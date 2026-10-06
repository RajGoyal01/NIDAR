"""Run with python -m nidar_survivor_demo.main or python path/to/main.py."""
from pathlib import Path
from collections import deque
from dataclasses import replace, asdict
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import logging
import time
from nidar_survivor_demo.config import arguments
from nidar_survivor_demo.camera.phone_stream import PhoneStream
from nidar_survivor_demo.preview import TITLE, render
from nidar_survivor_demo.utils.fps import FPSCounter
from nidar_survivor_demo.detector import DetectorConfig, PersonDetector, annotate
from nidar_survivor_demo.hybrid_detector import HybridPersonDetector
from nidar_survivor_demo.tracking import TrackerConfig, PersonTracker, annotate_tracks
from nidar_survivor_demo.verification import VerificationConfig, TemporalVerifier, annotate_verification
from nidar_survivor_demo.reid import ReIDConfig, AppearanceMatcher, annotate_reid
from nidar_survivor_demo.reid_encoder import AppearanceEncoder
from nidar_survivor_demo.survivors import SurvivorManager, ManagerConfig, new_mission_path, annotate_survivors
import cv2
import numpy as np


def run(argv: list[str] | None = None) -> int:
    config, args = arguments(argv)
    import torch
    import psutil
    process = psutil.Process()
    process.cpu_percent()
    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(args.cpu_threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    phase = 7 if args.dashboard else 6 if args.manage else 5 if args.reid else 4 if args.verify else 3 if args.track else 2 if args.detect else 1
    logging.info("Phase %s; source_type=%s", phase, "network" if config.network else "file" if config.file else "device")
    detector = None
    tracker = None
    verifier = None
    matcher = None
    if args.reid:
        try:
            reid_config = ReIDConfig.load(args.reid_config)
            matcher = AppearanceMatcher(reid_config, AppearanceEncoder(args.reid_model, args.device), retain_gallery=args.manage)
        except Exception as exc:
            logging.error("Re-ID startup failed (%s). Check Re-ID JSON, verified weights and device; run setup_reid.", type(exc).__name__)
            return 1
        print(json.dumps({"event": "reid_ready", "configuration": asdict(reid_config),
                          "device": matcher.encoder.device, "identity_scope": "session_appearance_reference",
                          "unique_counting": False}), flush=True)
    if args.verify:
        try:
            verifier = TemporalVerifier(VerificationConfig.load(args.verification_config))
        except ValueError as exc:
            logging.error("Verification configuration failed: %s", exc)
            return 1
        print(json.dumps({"event": "verification_ready", "configuration": asdict(verifier.config),
                          "meaning": "temporally verified person, not medical status", "unique_counting": False}), flush=True)
    if args.track:
        try:
            tracker_config = TrackerConfig.load(args.tracker_config)
            if args.confidence > tracker_config.track_low_thresh:
                raise ValueError("Detector confidence must not exceed tracker low threshold; low-score recovery needs those boxes.")
            tracker = PersonTracker(tracker_config)
        except ValueError as exc:
            logging.error("Tracker configuration failed: %s", exc)
            return 1
        except Exception as exc:
            logging.error("Tracker startup failed (%s); check pinned lap/Ultralytics requirements.", type(exc).__name__)
            return 1
        print(json.dumps({"event": "tracker_ready", "type": "botsort", "reid": False,
                          "configuration": asdict(tracker_config), "identity_scope": "temporary"}), flush=True)
    if args.detect:
        try:
            detector_config = DetectorConfig(args.model, args.imgsz, args.confidence,
                                             args.iou, args.device, args.fp32)
            detector = (HybridPersonDetector(detector_config, args.specialist_model, args.fusion_config)
                        if args.hybrid_detector else PersonDetector(detector_config))
        except Exception as exc:
            logging.error("Detector startup failed (%s). Check local model, CUDA and configuration.", type(exc).__name__)
            return 1
        print(json.dumps({"event": "detector_ready", "device": detector.device,
                          "half": detector.half, "imgsz": args.imgsz, "confidence": args.confidence,
                          "iou": args.iou, "class": getattr(detector, "class_name", "person"),
                          "model": str(Path(args.model).resolve()),
                          "mode": "validated_hybrid" if args.hybrid_detector else "single_model",
                          "specialist_model": (str(Path(args.specialist_model).resolve())
                                               if args.hybrid_detector else None),
                          "fusion_config": (str(Path(args.fusion_config).resolve())
                                            if args.hybrid_detector else None),
                          "unique_counting": False}), flush=True)
    stream = PhoneStream(config)
    displayed = FPSCounter()
    last_sequence = 0
    received = 0
    started = last_log = time.monotonic()
    next_preview = last_preview = 0.0
    last_state = None
    render_times: deque[float] = deque(maxlen=300)
    skipped = 0
    error = False
    detection = None
    processed_snapshot = None
    inference_times: deque[float] = deque(maxlen=300)
    tracking_times: deque[float] = deque(maxlen=300)
    tracking = None
    verification = None
    reidentification = None
    reid_times: deque[float] = deque(maxlen=300)
    reid_crops = 0
    verification_times: deque[float] = deque(maxlen=300)
    track_log = None
    manager = None
    dashboard = dashboard_server = None
    dashboard_events = []
    last_dashboard = 0.0
    managed = None
    manager_times: deque[float] = deque(maxlen=300)
    detection_status = "Phase 2: waiting for detection | Not survivor verification" if detector else None
    try:
        if args.manage:
            manager = SurvivorManager(args.mission_db or new_mission_path(), matcher,
                                      ManagerConfig.load(args.manager_config), resume=args.resume_mission)
            print(json.dumps({"event": "manager_ready", "mission_id": manager.mission_id,
                              "database": str(manager.path), "persistent_embeddings": True,
                              "count_meaning": "appearance-based estimated person records, not medical status"}), flush=True)
        if args.dashboard:
            from nidar_survivor_demo.dashboard import DashboardState, DashboardServer, annotate_dashboard
            dashboard = DashboardState("LOCAL VIDEO" if config.file else "LIVE", stale_after=config.stale_after,
                                       preview_fps=args.dashboard_fps, allow_reset=True)
            from nidar_survivor_demo.mission_report import read_report
            report = read_report(manager.path)
            dashboard.publish(status="STARTING", counts=report, mission=report, events=report["recent_events"],
                              modules={key: "WAITING" for key in ("Camera", "YOLO", "Tracker", "Re-ID", "Database")})
            dashboard_server = DashboardServer(dashboard, args.dashboard_port)
            print(json.dumps({"event": "dashboard_ready", "url": dashboard_server.url,
                              "controls": ["confirmed_new_mission"]}), flush=True)
        if args.track_log:
            args.track_log.parent.mkdir(parents=True, exist_ok=True)
            track_log = args.track_log.open("x", encoding="utf-8", buffering=1)
            track_log.write(json.dumps({"event": "tracking_session", "config": asdict(tracker_config),
                                        "id_scope": "temporary", "reid": False,
                                        "verification_config": asdict(verifier.config) if verifier else None,
                                        "reid_config": asdict(matcher.config) if matcher else None}) + "\n")
        if not args.headless:
            cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(TITLE, 1280, 720)
        stream.start()
        while not args.seconds or time.monotonic() - started < args.seconds:
            if dashboard and dashboard.take_reset():
                from nidar_survivor_demo.mission_control import rotate_mission
                manager, matcher, tracker, verifier, previous = rotate_mission(manager, tracker, verifier)
                detection = tracking = verification = reidentification = managed = processed_snapshot = None
                dashboard_events.clear()
                displayed = FPSCounter()
                for timings in (inference_times, tracking_times, verification_times, reid_times, manager_times, render_times):
                    timings.clear()
                reid_crops = 0
                fence = stream.snapshot()
                last_sequence = fence.packet.sequence if fence.packet else last_sequence
                dashboard.publish(status="STARTING", mission=manager.snapshot(),
                    counts={"current_persons": 0, "resolved_visible": 0, "pending_persons": 0,
                            "unique_survivors": 0, "duplicates_prevented": 0},
                    events=[{"event": "mission_started", "reason": "operator_reset"}], metrics={}, modules={})
                dashboard.complete_reset(previous)
                last_dashboard = time.monotonic()
                print(json.dumps({"event": "mission_reset", "previous_mission_id": previous["mission_id"],
                                  "mission_id": manager.mission_id, "database": str(manager.path)}), flush=True)
                continue
            iteration_started = time.perf_counter()
            snapshot = stream.snapshot()
            fresh = snapshot.packet is not None and snapshot.packet.sequence != last_sequence
            preview_due = iteration_started >= next_preview
            consume = fresh and (args.headless or preview_due)
            if consume:
                skipped += max(0, snapshot.packet.sequence - last_sequence - 1)
                last_sequence = snapshot.packet.sequence
                received += 1
                if detector:
                    detection = detector.predict(snapshot.packet.frame)
                    inference_times.append(detection.inference_ms)
                    if tracker:
                        tracking = tracker.update(detection, snapshot.packet.frame,
                                                  sequence=snapshot.packet.sequence,
                                                  received_at=snapshot.packet.received_at,
                                                  connection=snapshot.connections)
                        tracking_times.append(tracking.tracking_ms)
                        for event in tracking.events:
                            print(json.dumps(event), flush=True)
                        if track_log:
                            track_log.write(json.dumps({"event": "tracked_frame", "sequence": snapshot.packet.sequence,
                                                        "received_at": snapshot.packet.received_at,
                                                        "connection": snapshot.connections,
                                                        "tracking": asdict(tracking)}) + "\n")
                    annotated = snapshot.packet.frame
                    if verifier:
                        verification = None
                        # Gate evidence after inference: stale results must never confirm a track.
                        current = stream.snapshot()
                        if (current.packet is not None and current.connections == snapshot.connections
                                and time.monotonic() - snapshot.packet.received_at <= config.stale_after):
                            verification = verifier.update(tracking, sequence=snapshot.packet.sequence,
                                                           received_at=snapshot.packet.received_at,
                                                           epoch=tracker.epoch, frame_shape=snapshot.packet.frame.shape)
                            verification_times.append(verification.verification_ms)
                            for event in verification.events:
                                print(json.dumps(event), flush=True)
                            if track_log:
                                track_log.write(json.dumps({"event": "verified_frame", "sequence": snapshot.packet.sequence,
                                                            "verification": asdict(verification)}) + "\n")
                            if not dashboard:
                                annotated = annotate_verification(snapshot.packet.frame, verification, verifier.config)
                            if matcher:
                                def still_fresh():
                                    now = stream.snapshot()
                                    return (now.packet is not None and now.connections == snapshot.connections
                                            and time.monotonic() - snapshot.packet.received_at <= config.stale_after)
                                reidentification = matcher.update(snapshot.packet.frame, verification,
                                                                 sequence=snapshot.packet.sequence,
                                                                 received_at=snapshot.packet.received_at,
                                                                 epoch=tracker.epoch, is_fresh=still_fresh)
                                reid_times.append(reidentification.reid_ms)
                                reid_crops += reidentification.crops_encoded
                                for event in reidentification.events:
                                    print(json.dumps(event), flush=True)
                                if track_log:
                                    track_log.write(json.dumps({"event": "reid_frame", "sequence": snapshot.packet.sequence,
                                                                "reid": asdict(reidentification)}) + "\n")
                                if not dashboard:
                                    annotated = annotate_reid(annotated, verification, reidentification)
                                if manager:
                                    managed = manager.update(verification, reidentification, sequence=snapshot.packet.sequence,
                                                             received_at=snapshot.packet.received_at,
                                                             fresh=still_fresh() and not any(e["event"] == "reid_stale_discard" for e in reidentification.events))
                                    manager_times.append(managed.manager_ms)
                                    for event in managed.events:
                                        print(json.dumps(event), flush=True)
                                    if dashboard:
                                        dashboard_events.extend(managed.events)
                                    if not dashboard:
                                        annotated = annotate_survivors(annotated, verification, managed)
                                    if dashboard:
                                        annotated = annotate_dashboard(snapshot.packet.frame, verification, managed, reidentification)
                    else:
                        annotated = annotate_tracks(snapshot.packet.frame, tracking) if tracker else annotate(snapshot.packet.frame, detection)
                    packet = replace(snapshot.packet, frame=annotated)
                    processed_snapshot = replace(snapshot, packet=packet)
                displayed.tick()
            if detector:
                # Never put results on a different camera frame or retain boxes through an outage.
                health = stream.snapshot()
                age = (time.monotonic() - processed_snapshot.packet.received_at) if processed_snapshot else float("inf")
                wrong_connection = processed_snapshot is not None and processed_snapshot.connections != health.connections
                if health.packet is None or age > config.stale_after or wrong_connection:
                    snapshot = replace(health, packet=None, age_ms=None,
                                       state="STALE" if health.packet is not None else health.state)
                    detection = None
                    processed_snapshot = None
                    tracking = None
                    verification = None
                    reidentification = None
                    if manager:
                        managed = manager.unavailable()
                        if dashboard:
                            dashboard_events.extend(managed.events)
                    if matcher:
                        matcher.clear_tracks()
                    if verifier:
                        for reset_event in verifier.clear():
                            print(json.dumps(reset_event), flush=True)
                            if track_log:
                                track_log.write(json.dumps(reset_event) + "\n")
                    if tracker:
                        event = tracker.clear("camera_or_result_unavailable")
                        if event:
                            print(json.dumps(event), flush=True)
                            if track_log:
                                track_log.write(json.dumps(event) + "\n")
                elif processed_snapshot:
                    snapshot = replace(processed_snapshot, capture_fps=health.capture_fps, age_ms=age * 1000)
                else:
                    snapshot = replace(health, packet=None, age_ms=None)
                detection_status = (f"Phase 2 | People in THIS frame: {len(detection.people)} | AI total: {detection.inference_ms:.1f} ms | Not unique survivors"
                                    if detection else "Phase 2 | No fresh detection | Not survivor verification")
                if tracker:
                    detection_status = (f"Phase 3 | Visible tracks: {len(tracking.tracks)} | YOLO: {detection.inference_ms:.1f} ms | Track: {tracking.tracking_ms:.1f} ms | Temporary IDs, NOT unique survivors"
                                        if tracking and detection else "Phase 3 | No fresh tracks | Re-ID OFF | Temporary IDs only")
                if verifier:
                    detection_status = (f"Phase 4 | Verified visible: {verification.confirmed_count} | Verifying: {len(verification.tracks) - verification.confirmed_count} | {verifier.config.required}/{verifier.config.window} evidence | NOT unique / medical status"
                                        if verification else "Phase 4 | No fresh evidence | Unique counting OFF")
                if matcher:
                    detection_status = (f"Phase 5 | Verified: {verification.confirmed_count} | R IDs = session appearance, NOT unique survivors | Re-ID crops: {reid_crops}"
                                        if verification else "Phase 5 | No fresh evidence | Appearance gallery retained in RAM")
                if manager:
                    counts = managed or manager.unavailable()
                    detection_status = (f"Phase 6 | Current: {counts.current_persons if counts.current_persons is not None else '?'} | Pending: {counts.pending_persons if counts.pending_persons is not None else '?'} | Unique ESTIMATE: {counts.unique_survivors} | Duplicate events: {counts.duplicates_prevented}")
            rate = displayed.value() if snapshot.state == "ONLINE" else 0.0
            if dashboard and (time.monotonic() - last_dashboard >= 1 / args.dashboard_fps or snapshot.state != "ONLINE" and last_state == "ONLINE"):
                counts = managed or manager.unavailable()
                dashboard.publish(status=snapshot.state, counts=asdict(counts), mission=manager.snapshot(),
                    events=dashboard_events, frame=snapshot.packet.frame if snapshot.packet else None,
                    frame_at=snapshot.packet.received_at if snapshot.packet else None,
                    metrics={"capture_fps": snapshot.capture_fps, "processed_fps": rate,
                             "inference_ms": detection.inference_ms if detection else None,
                             "decoded_age_ms": snapshot.age_ms},
                    modules={"Camera": snapshot.state, "YOLO": "ACTIVE" if detection else "WAITING",
                             "Tracker": "ACTIVE" if tracking else "WAITING", "Re-ID": "ACTIVE" if reidentification else "WAITING",
                             "Database": "ACTIVE"})
                dashboard_events.clear()
                last_dashboard = time.monotonic()
            if not args.headless:
                # Sample faster than the source without drawing duplicate frames.
                # Health changes are immediate; idle metrics refresh four times/s.
                if consume or snapshot.state != last_state or iteration_started - last_preview >= 0.25:
                    render_started = time.perf_counter()
                    cv2.imshow(TITLE, render(snapshot, rate, detection_status) if detector else render(snapshot, rate))
                    render_times.append((time.perf_counter() - render_started) * 1000)
                    last_preview = iteration_started
                    next_preview = iteration_started + 1 / args.display_fps
                    last_state = snapshot.state
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
                    break
            elif not consume:
                time.sleep(0.001)
            if time.monotonic() - last_log >= 1:
                print(json.dumps({"state": snapshot.state, "capture_fps": round(snapshot.capture_fps, 2),
                                  "process_ram_mb": round(process.memory_info().rss / 1048576, 1),
                                  "process_cpu_percent": process.cpu_percent(),
                                  "cuda_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1048576, 1) if torch.cuda.is_available() else None,
                                  "dashboard_encode_ms": round(dashboard.encode_ms, 2) if dashboard else None,
                                  "consumer_fps" if args.headless else "display_fps": round(rate, 2),
                                  "decoded_age_ms": round(snapshot.age_ms, 1) if snapshot.age_ms is not None else None,
                                  "frames": snapshot.frames, "connections": snapshot.connections,
                                  "actual_size": list(snapshot.packet.frame.shape[1::-1]) if snapshot.packet else None,
                                  "consumer_skipped_frames": skipped,
                                  "people_in_frame": (len(tracking.tracks) if tracking else None) if tracker else (len(detection.people) if detection else None),
                                  "detection_candidates": len(detection.people) if detection else None,
                                  "visible_track_ids": [t.track_id for t in tracking.tracks] if tracking else None,
                                  "tracker_resets": tracker.resets if tracker else None,
                                  "track_episodes_started": tracker.started_tracks if tracker else None,
                                  "id_switches": None,  # Requires ground-truth identities, not ID churn.
                                  "verified_visible": verification.confirmed_count if verification else None,
                                  "verifying_visible": len(verification.tracks) - verification.confirmed_count if verification else None,
                                  "verification_p95_ms": round(float(np.percentile(verification_times, 95)), 3) if verification_times else None,
                                  "reid_p95_ms": round(float(np.percentile(reid_times, 95)), 3) if reid_times else None,
                                  "reid_crops_encoded": reid_crops,
                                  "appearance_observations": [asdict(o) for o in reidentification.observations] if reidentification else None,
                                  "survivor_counts": asdict(managed) if managed else None,
                                  "manager_p95_ms": round(float(np.percentile(manager_times, 95)), 3) if manager_times else None,
                                  "tracking_p95_ms": round(float(np.percentile(tracking_times, 95)), 2) if tracking_times else None,
                                  "inference_p95_ms": round(float(np.percentile(inference_times, 95)), 2) if inference_times else None,
                                  "render_p95_ms": round(float(np.percentile(render_times, 95)), 2) if render_times else None}), flush=True)
                last_log = time.monotonic()
            if snapshot.state == "STOPPED":
                logging.error("Capture worker stopped unexpectedly.")
                error = True
                break
            if not args.headless:
                # Short event-loop wait, independent of camera FPS. A full frame
                # sleep here can miss the arrival of the next camera frame.
                time.sleep(0.001)
    except KeyboardInterrupt:
        logging.info("Ctrl+C requested shutdown")
    except cv2.error:
        logging.error("OpenCV window error; confirm GUI-enabled OpenCV and desktop access.")
        error = True
    except Exception as exc:
        logging.error("Processing failed (%s); stopping safely, not reporting zero people.", type(exc).__name__)
        error = True
    finally:
        if dashboard:
            dashboard.stop("ERROR" if error else "STOPPED")
        if dashboard_server:
            dashboard_server.close()
        clean = stream.stop()
        if manager:
            try:
                manager.close()
                print(json.dumps({"event": "mission_summary", **manager.snapshot()}), flush=True)
            except Exception as exc:
                logging.error("Mission persistence close failed (%s); inspect database before resume.", type(exc).__name__)
                error = True
        if not args.headless:
            cv2.destroyAllWindows()
        if track_log:
            track_log.close()
        print(json.dumps({"event": "shutdown", "clean": clean, "unique_frames_consumed": received}), flush=True)
    return 1 if error or not clean or received == 0 else 0


if __name__ == "__main__":
    raise SystemExit(run())
