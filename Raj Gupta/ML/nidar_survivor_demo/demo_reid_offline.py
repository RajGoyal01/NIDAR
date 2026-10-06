"""Camera-free labelled replay using real COCO crops, real OSNet and temporal logic.

Synthetic motion/brightness are NOT independent live re-entry views. Oracle boxes
isolate Phase 5 from detector accuracy; reference truth comes from COCO annotations.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .reid import AppearanceMatcher, ReIDConfig, annotate_reid
from .reid_encoder import AppearanceEncoder
from .tracking import Track, TrackingResult
from .verification import TemporalVerifier, VerificationConfig, annotate_verification
from .survivors import SurvivorManager, ManagerConfig, annotate_survivors

ROOT = Path(__file__).resolve().parents[1]


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "logs/phase5_coco_proxy.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--show", action="store_true", help="Display known-input replay; no phone required.")
    parser.add_argument("--pairs", type=int, choices=range(1, 21), default=10)
    parser.add_argument("--visible-frames", type=int, choices=range(16, 201), default=16,
                        help="Processed frames per visible stage at simulated 10 FPS; use 60 for sustained novelty.")
    parser.add_argument("--manage-dir", type=Path, help="Phase 6: new per-pair mission databases and resume validation.")
    parser.add_argument("--dashboard", action="store_true", help="Show the real-model offline replay through Phase 7.")
    parser.add_argument("--dashboard-port", type=int, default=8765)
    parser.add_argument("--hold-dashboard", type=float, default=0, help="Keep final report visible for N seconds.")
    args = parser.parse_args()
    if args.dashboard and not args.manage_dir:
        parser.error("--dashboard requires --manage-dir")
    if not 0 <= args.hold_dashboard <= 86400:
        parser.error("--hold-dashboard must be between 0 and 86400")
    manifest = json.loads(args.manifest.read_text())
    pairs = manifest["records"][20:20 + args.pairs]  # Held-out images only.
    annotations = json.loads((ROOT / "datasets/coco2017/annotations/instances_val2017.json").read_text())
    annotations = {str(a["id"]): a for a in annotations["annotations"]}
    encoder = AppearanceEncoder()
    config = ReIDConfig.load(ROOT / "nidar_survivor_demo/settings/reid.json")
    events, results, times = [], [], []
    total_encoded = total_frames = 0
    manager = None
    manager_times = []
    dashboard = server = None
    try:
        if args.dashboard:
            from .dashboard import DashboardState, DashboardServer
            dashboard = DashboardState("OFFLINE REPLAY")
            server = DashboardServer(dashboard, args.dashboard_port)
            print(json.dumps({"dashboard_url": server.url, "mode": "OFFLINE REPLAY"}), flush=True)
        for pair in pairs:
            image = cv2.imread(str(ROOT / "datasets/coco2017/images/val2017" / f'{pair["image_id"]:012d}.jpg'))
            crops = []
            for label in pair["annotation_ids"]:
                x, y, w, h = annotations[label]["bbox"]
                crops.append(image[max(0, int(y)):min(image.shape[0], int(np.ceil(y + h))),
                                   max(0, int(x)):min(image.shape[1], int(np.ceil(x + w)))])
            matcher = AppearanceMatcher(config, encoder, retain_gallery=bool(args.manage_dir))
            if args.manage_dir:
                manager = SurvivorManager(args.manage_dir / f'pair-{pair["image_id"]}.sqlite3', matcher,
                                          ManagerConfig.load(ROOT / "nidar_survivor_demo/settings/survivors.json"))
            verifier = TemporalVerifier(VerificationConfig())
            sequence = 0
            # Identity IDs are annotation-local, not inferred from visual appearance.
            stages = [("A_enroll", 0), ("empty", None), ("A_return", 0), ("empty", None),
                      ("B_enroll", 1), ("empty", None), ("B_return", 1)]
            if manager:
                stages = [("A_enroll", 0), ("empty", None), ("B_enroll", 1), ("empty", None),
                          ("A_return", 0), ("empty", None), ("B_return", 1)]
            refs = {}
            manager_stages = {}
            resumed_count = None
            for stage_index, (stage, person) in enumerate(stages):
                last = None
                for index in range(args.visible_frames if person is not None else 12):
                    sequence += 1
                    frame = np.full((480, 640, 3), 40, np.uint8)
                    tracks = ()
                    if person is not None:
                        crop = cv2.resize(crops[person], (128, 300))
                        if "return" in stage:
                            crop = cv2.convertScaleAbs(cv2.flip(crop, 1), alpha=.93, beta=6)
                        x, y = 240 + index // 4, 80
                        frame[y:y + 300, x:x + 128] = crop
                        tracks = (Track(f"E1:T{stage_index + 1}", (x, y, x + 128, y + 300), .95),)
                    verified = verifier.update(TrackingResult(tracks, 0, ()), sequence=sequence,
                                               received_at=sequence * .1, epoch=1, frame_shape=frame.shape)
                    start = time.perf_counter()
                    result = matcher.update(frame, verified, sequence=sequence, received_at=sequence * .1, epoch=1)
                    managed = None
                    if manager:
                        managed = manager.update(verified, result, sequence=sequence, received_at=sequence * .1)
                        manager_times.append(managed.manager_ms)
                        for event in managed.events:
                            events.append({"image_id": pair["image_id"], "stage": stage, **event})
                    times.append((time.perf_counter() - start) * 1000)
                    total_frames += 1
                    total_encoded += result.crops_encoded
                    for event in result.events:
                        events.append({"image_id": pair["image_id"], "stage": stage, **event})
                    last = result.observations[0] if result.observations else None
                    if args.show or dashboard:
                        rendered = annotate_reid(annotate_verification(frame, verified, verifier.config), verified, result)
                        if managed:
                            rendered = annotate_survivors(rendered, verified, managed)
                            cv2.putText(rendered, f"UNIQUE ESTIMATE {managed.unique_survivors} | DUPLICATES {managed.duplicates_prevented} | PENDING {managed.pending_persons}",
                                        (8, 455), cv2.FONT_HERSHEY_SIMPLEX, .5, (80, 240, 240), 1)
                        cv2.putText(rendered, f"OFFLINE LABELLED REPLAY | {stage} | oracle boxes", (8, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1)
                        if dashboard:
                            dashboard.publish(status="ONLINE", counts=asdict(managed), mission=manager.snapshot(),
                                events=managed.events, frame=rendered, stage=stage,
                                metrics={"capture_fps": None, "processed_fps": None, "inference_ms": None, "decoded_age_ms": None},
                                modules={"Camera": "REPLAY", "YOLO": "ORACLE BOXES", "Tracker": "ORACLE TRACKS", "Re-ID": "ACTIVE", "Database": "ACTIVE"})
                            time.sleep(.2)
                        if args.show:
                            cv2.imshow("NIDAR Phase 5 offline test", rendered)
                            if cv2.waitKey(50) & 255 in (27, ord("q")):
                                raise KeyboardInterrupt
                if person is not None:
                    refs[stage] = asdict(last) if last else None
                    if manager:
                        manager_stages[stage] = asdict(managed)
                if manager and stage_index == 3:
                    # Simulate application restart with fresh manager/matcher objects.
                    database, manager_config = manager.path, manager.config
                    before = manager.snapshot()["unique_survivors"]
                    manager.close()
                    matcher = AppearanceMatcher(config, encoder, retain_gallery=True)
                    manager = SurvivorManager(database, matcher, manager_config, resume=True)
                    resumed_count = manager.snapshot()["unique_survivors"]
                    if before != resumed_count:
                        raise AssertionError("Resume changed unique count.")
                    verifier = TemporalVerifier(VerificationConfig())
                    sequence = 0
            def ref(stage):
                return refs[stage]["reference_id"] if refs[stage] else None
            false_merge = any(ref(s) and ref(s) == ref("A_enroll") for s in ("B_enroll", "B_return"))
            correct_returns = sum(ref(s + "_enroll") is not None and ref(s + "_return") == ref(s + "_enroll") for s in ("A", "B"))
            false_splits = sum(ref(s + "_enroll") is not None and ref(s + "_return") is not None
                               and ref(s + "_return") != ref(s + "_enroll") for s in ("A", "B"))
            results.append({"image_id": pair["image_id"], "annotation_ids": pair["annotation_ids"],
                            "observations": refs, "false_merge": bool(false_merge),
                            "correct_returns": correct_returns, "false_splits": false_splits})
            if manager:
                for stage in ("A_return", "B_return"):
                    if manager_stages[stage]["unique_survivors"] != resumed_count:
                        raise AssertionError("Repeat appearance inflated persistent count.")
                for person_name in ("A", "B"):
                    enrolled = list(manager_stages[f"{person_name}_enroll"]["track_survivors"].values())
                    returned = list(manager_stages[f"{person_name}_return"]["track_survivors"].values())
                    if enrolled and returned != enrolled:
                        raise AssertionError("Known enrolled survivor ID changed on return.")
                results[-1]["manager_stages"] = manager_stages
                results[-1]["resume_unique_count"] = resumed_count
                results[-1]["mission_summary"] = manager.snapshot()
                manager.close()
                manager = None
        report = {"scope": "Offline synthetic replay of real labelled crops; oracle tracks, actual OSNet/temporal/matcher; NOT live accuracy",
                  "configuration": asdict(config), "pairs": len(results), "return_attempts": 2 * len(results),
                  "correct_returns": sum(r["correct_returns"] for r in results),
                  "false_merge_pairs": sum(r["false_merge"] for r in results),
                  "false_splits": sum(r["false_splits"] for r in results),
                  "unresolved_returns": sum(r["observations"][s]["reference_id"] is None for r in results for s in ("A_return", "B_return")),
                  "frames": total_frames, "crops_encoded": total_encoded,
                  "loop_median_ms": float(np.median(times)), "loop_p95_ms": float(np.percentile(times, 95)),
                  "results": results, "events": events}
        if args.manage_dir:
            report["manager_validation"] = {
                "resumed_missions": len(results), "return_count_inflations": 0,
                "survivor_records": sum(r["mission_summary"]["unique_survivors"] for r in results),
                "duplicate_events": sum(r["mission_summary"]["duplicates_prevented"] for r in results),
                "manager_median_ms": float(np.median(manager_times)),
                "manager_p95_ms": float(np.percentile(manager_times, 95))}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as out:
            json.dump(report, out, indent=2)
        print(json.dumps({k: v for k, v in report.items() if k not in ("results", "events")}, indent=2))
        if dashboard:
            dashboard.publish(status="COMPLETE", events=[{"event": "replay_complete"}])
            deadline = time.monotonic() + args.hold_dashboard
            try:
                while time.monotonic() < deadline:
                    time.sleep(.25)
            except KeyboardInterrupt:
                pass  # Report already saved; Ctrl+C ends only the review server.
    finally:
        if server:
            server.close()
        if manager:
            manager.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
