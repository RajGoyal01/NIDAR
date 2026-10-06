"""Loopback dashboard with explicit guarded reset. ML owns identity decisions."""
import argparse
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import secrets
import threading
import time
from urllib.parse import urlsplit

import cv2
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
import uvicorn

ASSETS = Path(__file__).with_name("web")
COUNTS = ("current_persons", "resolved_visible", "pending_persons", "unique_survivors", "duplicates_prevented")
RECORD_FIELDS = ("survivor_id", "reference_id", "status", "first_seen_at", "last_seen_at", "current_track_id", "latest_camera_confidence", "identity_score")
EVENT_FIELDS = ("event", "survivor_id", "reference_id", "track_id", "reason", "score", "margin", "sequence", "recorded_at")


PENDING_GUIDANCE = {
    "partial_view_needs_full_evidence": "Partial view: existing identity uncertain",
    "frame_edge_crop": "Move back: keep body inside frame",
    "clipped_crop": "Move back: body is cut off",
    "low_confidence": "Improve light / straighten camera",
    "small_crop": "Move closer: person too small",
    "blurred_crop": "Hold camera steady",
    "overlapping_person": "Separate overlapping people",
    "need_good_views": "Collecting clear views",
    "confirming_new_person": "Checking new person",
    "inconsistent_views": "Appearance changing: hold steady",
    "similarity_or_margin_ambiguous": "Identity uncertain: show clearer view",
}


def annotate_dashboard(frame, verified, managed, reidentified=None):
    """One readable label per track, without changing tracker/identity decisions."""
    output = frame.copy()
    h, w = output.shape[:2]
    reasons = {o.track_id: o.reason for o in reidentified.observations} if reidentified else {}
    for item in verified.tracks:
        x1, y1, x2, y2 = [int(v) for v in item.track.xyxy]
        x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
        y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
        sid = managed.track_survivors.get(item.track.track_id)
        label = f"{sid} | person {item.track.confidence:.0%}" if sid else f"{item.track.track_id} | identity pending"
        if not sid and item.track.track_id in reasons:
            reason = reasons[item.track.track_id]
            label = f"{item.track.track_id} | {PENDING_GUIDANCE.get(reason, 'Checking identity')}"
        if item.state != "CONFIRMED":
            label = f"Person candidate {item.track.confidence:.0%} | verifying ({item.hits} good frames)"
        color = (120, 230, 180) if sid else (100, 185, 235)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .48, 1)[0][0] + 12
        tx = max(0, min(x1, w - width))
        ty = max(22, y1)
        cv2.rectangle(output, (tx, ty - 22), (min(w - 1, tx + width), ty), (25, 33, 34), -1)
        cv2.putText(output, label, (tx + 6, ty - 7), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)
    return output


class DashboardState:
    """Bounded, thread-safe latest-value mailbox; never queues frames."""
    def __init__(self, mode="LIVE", stale_after=1.5, preview_fps=30, allow_reset=False):
        if type(preview_fps) is not int or not 1 <= preview_fps <= 60:
            raise ValueError("Preview FPS must be an integer from 1 to 60")
        self.preview_fps = preview_fps
        self.encode_ms = 0.0
        self._encoded_frame_at = None
        if mode not in {"LIVE", "LOCAL VIDEO", "OFFLINE REPLAY", "ARCHIVE"}:
            raise ValueError("Invalid dashboard mode")
        self.mode, self.stale_after = mode, stale_after
        self._lock = threading.Lock()
        self.allow_reset = allow_reset and mode in {"LIVE", "LOCAL VIDEO"}
        self.reset_token = secrets.token_urlsafe(32)
        self._reset_pending = False
        self._reset_taken = False
        self._archives = deque(maxlen=20)
        self._updated = self._frame_at = self._last_encode = 0.0
        self._jpeg = None
        self._events = deque(maxlen=100)
        self._event_id = 0
        self._data = {"mode": mode, "status": "STARTING", "mission_id": None,
                      "counts": dict.fromkeys(COUNTS), "records": [], "metrics": {}, "modules": {},
                      "stage": None, "frame_sequence": 0}

    def publish(self, *, status, counts=None, mission=None, events=(), metrics=None,
                modules=None, frame=None, frame_at=None, stage=None):
        now = time.monotonic()
        jpeg = None
        if (frame is not None and now - self._last_encode >= 1 / self.preview_fps
                and (frame_at is None or frame_at != self._encoded_frame_at)):
            # Encode only a new processed frame. No duplicate work or frame queue.
            encode_started = time.perf_counter()
            h, w = frame.shape[:2]
            resized = cv2.resize(frame, (960, max(1, round(h * 960 / w)))) if w > 960 else frame
            ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                jpeg = encoded.tobytes()
                self._last_encode = now
                self._encoded_frame_at = frame_at
                self.encode_ms = (time.perf_counter() - encode_started) * 1000
        with self._lock:
            self._data.update(status=status, stage=stage)
            if counts is not None:
                self._data["counts"] = {key: counts.get(key) for key in COUNTS}
            if mission is not None:
                if self._data["mission_id"] != mission.get("mission_id"):
                    self._events.clear()
                self._data["mission_id"] = mission.get("mission_id")
                self._data["records"] = [{k: r.get(k) for k in RECORD_FIELDS} for r in mission.get("records", [])[:64]]
            if metrics is not None:
                self._data["metrics"] = deepcopy(metrics)
            if modules is not None:
                self._data["modules"] = deepcopy(modules)
            for event in events:
                self._event_id += 1
                safe = {k: event[k] for k in EVENT_FIELDS if k in event}
                safe.update(id=self._event_id)
                safe.setdefault("recorded_at", datetime.now(timezone.utc).isoformat())
                self._events.append(safe)
            if jpeg is not None:
                self._jpeg = jpeg
                self._frame_at = frame_at if frame_at is not None else now
                self._data["frame_sequence"] += 1
            if status != "ONLINE":
                self._jpeg = None
            self._updated = now

    def snapshot(self):
        with self._lock:
            data = deepcopy(self._data)
            data["reset"] = {"available": self.allow_reset and data["status"] not in {"STOPPED", "ERROR"},
                             "pending": self._reset_pending, "token": self.reset_token if self.allow_reset else None}
            data["previous_missions"] = [{"mission_id": r["mission_id"], "unique_survivors": r["unique_survivors"]} for r in self._archives]
            data["events"] = list(deepcopy(self._events))
            data["preview_fps_limit"] = self.preview_fps
            data["metrics"]["jpeg_encode_ms"] = self.encode_ms
            age = time.monotonic() - self._updated if self._updated else None
            stale = self.mode != "ARCHIVE" and (age is None or age > self.stale_after)
            if stale and data["status"] == "ONLINE":
                data["status"] = "STALE"
            data["update_age_ms"] = round(age * 1000) if age is not None else None
            data["frame_available"] = bool(self._jpeg and not stale and data["status"] == "ONLINE"
                                           and time.monotonic() - self._frame_at <= self.stale_after)
            if data["status"] != "ONLINE":
                for key in COUNTS[:3]:
                    data["counts"][key] = None
                for record in data["records"]:
                    record["status"] = "NOT_VISIBLE" if self.mode == "ARCHIVE" else "UNKNOWN"
                    record["current_track_id"] = None
                data["modules"] = {key: "ARCHIVE" if self.mode == "ARCHIVE" else data["status"] for key in data["modules"]}
                data["metrics"] = {key: None for key in data["metrics"]}
            return data

    def request_reset(self, mission_id):
        with self._lock:
            if not self.allow_reset or self._data["status"] in {"STOPPED", "ERROR"}:
                return 409
            if not mission_id or mission_id != self._data["mission_id"] or self._reset_pending:
                return 409
            self._reset_pending = True
            self._reset_taken = False
            return 202

    def take_reset(self):
        with self._lock:
            if not self._reset_pending or self._reset_taken:
                return False
            self._reset_taken = True
            return True

    def complete_reset(self, old_report):
        with self._lock:
            safe = {k: old_report[k] for k in ("mission_id", "unique_survivors", "duplicates_prevented", "meaning")}
            safe["records"] = [{k: r.get(k) for k in RECORD_FIELDS} for r in old_report["records"]]
            self._archives.append(safe)
            self._reset_pending = self._reset_taken = False

    def archive(self, mission_id):
        with self._lock:
            return next((deepcopy(r) for r in self._archives if r["mission_id"] == mission_id), None)

    def frame(self, with_sequence=False):
        with self._lock:
            if (self._data["status"] != "ONLINE" or not self._jpeg
                    or time.monotonic() - self._frame_at > self.stale_after):
                return None
            return (self._jpeg, self._data["frame_sequence"], self._data["mission_id"]) if with_sequence else self._jpeg

    def stop(self, status="STOPPED"):
        self.publish(status=status)


def create_app(state: DashboardState, port: int):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    @app.middleware("http")
    async def boundary(request: Request, call_next):
        origin = request.headers.get("origin")
        if (request.headers.get("host") not in hosts or
                (origin and origin not in {"http://" + h for h in hosts}) or
                request.headers.get("sec-fetch-site") == "cross-site"):
            return Response(status_code=403)
        reset_request = request.method == "POST" and request.url.path == "/api/mission/reset"
        if reset_request and (origin != "http://" + request.headers.get("host", "") or
                not secrets.compare_digest(request.headers.get("x-reset-token", ""), state.reset_token)):
            return Response(status_code=403)
        if request.method not in {"GET", "HEAD"} and not reset_request:
            return Response(status_code=405)
        response = await call_next(request)
        response.headers.update({"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                                 "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
                                 "Content-Security-Policy": "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"})
        return response

    @app.get("/")
    def index():
        return FileResponse(ASSETS / "index.html")

    @app.get("/app.js")
    def script():
        return FileResponse(ASSETS / "app.js", media_type="text/javascript")

    @app.get("/style.css")
    def stylesheet():
        return FileResponse(ASSETS / "style.css", media_type="text/css")

    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    @app.get("/api/state")
    def snapshot():
        return state.snapshot()

    @app.get("/api/frame")
    def frame():
        value = state.frame(with_sequence=True)
        return Response(value[0], media_type="image/jpeg", headers={"X-Frame-Sequence": str(value[1]), "X-Mission-ID": value[2] or ""}) if value else Response(status_code=204)

    @app.get("/api/report")
    def report():
        report = state.snapshot()
        report.pop("reset", None)
        return JSONResponse(report, headers={"Content-Disposition": 'attachment; filename="nidar-mission-report.json"'})

    @app.post("/api/mission/reset")
    def reset(request: Request):
        code = state.request_reset(request.headers.get("x-mission-id"))
        return JSONResponse({"status": "queued" if code == 202 else "mission_changed_or_reset_unavailable"}, status_code=code)

    @app.get("/api/archive/{mission_id}")
    def archive(mission_id: str):
        report = state.archive(mission_id)
        return JSONResponse(report, headers={"Content-Disposition": 'attachment; filename="previous-mission.json"'}) if report else Response(status_code=404)

    return app


class DashboardServer:
    """Bind before spawning: occupied ports fail visibly, with no silent fallback."""
    def __init__(self, state, port=8765):
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("Invalid dashboard port")
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            self.socket.bind(("127.0.0.1", port))
            self.socket.listen(32)
            self.port = self.socket.getsockname()[1]
            self.url = f"http://127.0.0.1:{self.port}"
            self.server = uvicorn.Server(uvicorn.Config(create_app(state, self.port), log_level="warning",
                 access_log=False, limit_concurrency=32, timeout_keep_alive=2, server_header=False))
            self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [self.socket]}, daemon=True)
            self.thread.start()
            deadline = time.monotonic() + 5
            while not self.server.started and self.thread.is_alive() and time.monotonic() < deadline:
                time.sleep(.01)
            if not self.server.started:
                raise RuntimeError("Dashboard server could not start")
        except Exception:
            self.socket.close()
            raise

    def close(self):
        self.server.should_exit = True
        self.thread.join(timeout=5)
        self.socket.close()


def main():
    parser = argparse.ArgumentParser(description="Read-only saved-mission dashboard (no camera required).")
    parser.add_argument("--mission", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    from .mission_report import read_report
    report = read_report(args.mission)
    state = DashboardState("ARCHIVE")
    state.publish(status="ARCHIVE", counts=report, mission=report, events=report["recent_events"],
                  modules={k: "ARCHIVE" for k in ("Camera", "YOLO", "Tracker", "Re-ID", "Database")})
    server = DashboardServer(state, args.port)
    print(json.dumps({"dashboard_url": server.url, "mode": "ARCHIVE"}), flush=True)
    try:
        while server.thread.is_alive():
            time.sleep(.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    main()
