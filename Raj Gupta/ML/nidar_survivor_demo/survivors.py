"""Mission-scoped identity authority with transactional, local SQLite persistence."""
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
import uuid
import cv2
from .reid import AppearanceMatcher, ReIDResult
from .setup_reid import SHA256
from .verification import VerificationResult

DEFAULT_MANAGER_CONFIG = Path(__file__).resolve().parent / "settings/survivors.json"
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ManagerConfig:
    new_identity_hits: int = 3
    observation_interval: float = 1.0
    max_observations_per_survivor: int = 100
    max_track_history: int = 100
    max_events: int = 100000

    def __post_init__(self):
        for name, upper in (("new_identity_hits", 30), ("max_observations_per_survivor", 1000),
                            ("max_track_history", 1000), ("max_events", 1000000)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"Invalid manager bound: {name}.")
        if type(self.observation_interval) not in (float, int) or not math.isfinite(self.observation_interval) or self.observation_interval <= 0:
            raise ValueError("Observation interval must be finite and positive.")

    @classmethod
    def load(cls, path: Path):
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError("Invalid survivor-manager JSON configuration.") from exc


@dataclass(frozen=True)
class ManagerResult:
    current_persons: int | None
    resolved_visible: int | None
    pending_persons: int | None
    unique_survivors: int
    duplicates_prevented: int
    track_survivors: dict[str, str]
    events: tuple[dict, ...]
    manager_ms: float


def new_mission_path() -> Path:
    return ROOT / "runs/missions" / str(uuid.uuid4()) / "mission.sqlite3"


class SurvivorManager:
    """Single writer. Database commits succeed before in-memory counts advance."""
    def __init__(self, path: Path, matcher: AppearanceMatcher, config: ManagerConfig = ManagerConfig(), *, resume: bool = False):
        if not matcher.retain_gallery:
            raise ValueError("Managed missions require retained appearance history.")
        self.path, self.matcher, self.config = Path(path), matcher, config
        self.run_id = str(uuid.uuid4())
        self._records = {}
        self._bindings = {}
        self._candidates = {}
        self._pending_states = {}
        self._last_sequence = self._last_time = None
        self._last_observation = {}
        self._duplicates = 0
        self._events_count = 0
        self._unavailable = True
        self._closed = False
        self._failed = False
        self._db = None
        self._lock = None
        signature = {"schema": 1, "encoder_sha256": SHA256, "embedding_dimension": 512,
                     "reid_config": asdict(matcher.config), "manager_config": asdict(config)}
        self.signature = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.path.with_suffix(self.path.suffix + ".writer-lock")
        # An explicit lock prevents two trackers from modifying the same mission.
        self._lock = self._lock_path.open("x", encoding="utf-8")
        self._lock.write(json.dumps({"run_id": self.run_id, "pid": os.getpid(), "created_at": datetime.now(timezone.utc).isoformat()}))
        self._lock.flush()
        try:
            if resume:
                if not self.path.is_file():
                    raise ValueError("Cannot resume a missing mission.")
                self._db = sqlite3.connect(self.path.resolve().as_uri() + "?mode=rw", uri=True, timeout=2)
                meta = dict(self._db.execute("SELECT key, value FROM metadata"))
                if meta.get("signature") != self.signature or meta.get("schema") != "1":
                    raise ValueError("Mission schema/model/settings differ; use a new mission.")
                if self._db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise ValueError("Mission integrity check failed.")
                self.mission_id = meta["mission_id"]
                self._duplicates = int(meta["duplicates_prevented"])
                self._events_count = self._db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                self._records = {row[0]: json.loads(row[1]) for row in self._db.execute("SELECT reference_id, record FROM survivors")}
                if self._duplicates < 0 or len(self._records) > matcher.config.max_gallery:
                    raise ValueError("Invalid mission counters/capacity.")
                expected_sids = {f"S{i}" for i in range(1, len(self._records) + 1)}
                if {record["survivor_id"] for record in self._records.values()} != expected_sids:
                    raise ValueError("Invalid survivor ID sequence.")
                for reference, record in self._records.items():
                    if (record["reference_id"] != reference or not isinstance(record["observations"], list)
                            or not isinstance(record["track_history"], list)
                            or len(record["observations"]) > config.max_observations_per_survivor
                            or len(record["track_history"]) > config.max_track_history):
                        raise ValueError("Invalid survivor history.")
                gallery = json.loads(meta["gallery"])
                if not set(self._records).issubset(gallery):
                    raise ValueError("Mission records lack appearance evidence.")
                if any(len(v) != 512 for vectors in gallery.values() for v in vectors):
                    raise ValueError("Mission embedding dimension mismatch.")
                matcher.restore_gallery(gallery)
            else:
                # Reserve exact new path; never overwrite an existing mission.
                with self.path.open("xb"):
                    pass
                self._db = sqlite3.connect(self.path, timeout=2)
                self._db.executescript("""
                    CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                    CREATE TABLE survivors(reference_id TEXT PRIMARY KEY, survivor_id TEXT UNIQUE NOT NULL, record TEXT NOT NULL);
                    CREATE TABLE events(id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
                """)
                self.mission_id = str(uuid.uuid4())
                with self._db:
                    self._db.executemany("INSERT INTO metadata VALUES (?, ?)",
                        [("schema", "1"), ("signature", self.signature), ("mission_id", self.mission_id),
                         ("duplicates_prevented", "0"), ("gallery", "{}")])
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            self._records = {key: {**value, "current_track_id": None, "status": "NOT_VISIBLE"} for key, value in self._records.items()}
            self._persist(self._records, self._duplicates, [{"event": "mission_resumed" if resume else "mission_started"}])
        except Exception:
            self._release()
            raise

    def _persist(self, records, duplicates, events):
        if self._failed or self._closed:
            raise RuntimeError("Mission store is closed or failed.")
        if self._events_count + len(events) > self.config.max_events:
            self._failed = True
            raise RuntimeError("Mission event limit reached; stop without discarding audit history.")
        stamp = datetime.now(timezone.utc).isoformat()
        payloads = [{"mission_id": self.mission_id, "run_id": self.run_id, "recorded_at": stamp, **event} for event in events]
        try:
            with self._db:
                for ref, record in records.items():
                    self._db.execute("INSERT INTO survivors VALUES (?, ?, ?) ON CONFLICT(reference_id) DO UPDATE SET record=excluded.record",
                                     (ref, record["survivor_id"], json.dumps(record, allow_nan=False)))
                self._db.executemany("INSERT INTO events(payload) VALUES (?)", [(json.dumps(e, allow_nan=False),) for e in payloads])
                self._db.execute("UPDATE metadata SET value=? WHERE key='duplicates_prevented'", (str(duplicates),))
                self._db.execute("UPDATE metadata SET value=? WHERE key='gallery'", (json.dumps(self.matcher.export_gallery(), allow_nan=False),))
        except Exception:
            self._failed = True
            raise
        self._events_count += len(events)

    def snapshot(self) -> dict:
        return {"mission_id": self.mission_id, "unique_survivors": len(self._records),
                "duplicates_prevented": self._duplicates, "meaning": "appearance-based estimated person records, not medical status",
                "records": json.loads(json.dumps(list(self._records.values())))}

    def unavailable(self, reason: str = "camera_or_result_unavailable") -> ManagerResult:
        if not self._unavailable:
            records = {key: {**record, "status": "NOT_VISIBLE", "current_track_id": None} for key, record in self._records.items()}
            events = ({"event": "perception_unavailable", "reason": reason},)
            self._persist(records, self._duplicates, events)
            self._records = records
        else:
            events = ()
        self._bindings.clear()
        self._candidates.clear()
        self._pending_states.clear()
        self._unavailable = True
        return ManagerResult(None, None, None, len(self._records), self._duplicates, {}, events, 0)

    def update(self, verified: VerificationResult, reid: ReIDResult, *, sequence: int,
               received_at: float, fresh: bool = True) -> ManagerResult:
        started = time.perf_counter()
        if not fresh:
            return self.unavailable("stale_result")
        if type(sequence) is not int or sequence < 1 or not math.isfinite(received_at):
            raise ValueError("Manager requires valid frame sequence/time.")
        if self._last_sequence is not None and (sequence <= self._last_sequence or received_at < self._last_time):
            raise ValueError("Manager requires chronological, unrepeated frames.")
        tracks = {v.track.track_id: v for v in verified.tracks}
        observations = {o.track_id: o for o in reid.observations}
        if len(tracks) != len(verified.tracks) or len(observations) != len(reid.observations) or not set(observations).issubset(tracks):
            raise ValueError("Mismatched/duplicate manager observations.")
        for item in verified.tracks:
            if not all(math.isfinite(x) for x in (*item.track.xyxy, item.track.confidence)) or not 0 <= item.track.confidence <= 1:
                raise ValueError("Invalid survivor observation.")
        for o in reid.observations:
            if any(value is not None and not math.isfinite(value) for value in (o.score, o.margin)):
                raise ValueError("Invalid identity evidence score.")
        confirmed = {key: value.track for key, value in tracks.items() if value.state == "CONFIRMED"}
        records = json.loads(json.dumps(self._records))  # Transactional working copy.
        candidates, bindings, pending_states = {}, {}, {}
        seen_refs = Counter(o.reference_id for o in reid.observations if o.track_id in confirmed and o.reference_id)
        events = []
        duplicates = self._duplicates
        output = {}
        stamp = datetime.now(timezone.utc).isoformat()
        last_observation = dict(self._last_observation)
        # Preserve binding continuity through weak evidence only while the track remains visible.
        old_bindings = {k: v for k, v in self._bindings.items() if k in tracks}
        for label, track in confirmed.items():
            o = observations.get(label)
            reference = o.reference_id if o else None
            reason = o.reason if o else "missing_reid_result"
            if reference and seen_refs[reference] > 1:
                reason, reference = "simultaneous_reference_conflict", None
            if reference and (reference not in self.matcher.gallery or o.state not in ("NEW_REFERENCE", "MATCHED")):
                reason, reference = "invalid_reference_evidence", None
            if reference and reference not in records:
                previous_ref, hits = self._candidates.get(label, (None, 0))
                hits = hits + 1 if previous_ref == reference else 1
                candidates[label] = (reference, hits)
                if hits < self.config.new_identity_hits:
                    reason, reference = "awaiting_identity_confirmation", None
                else:
                    sid = f"S{len(records) + 1}"
                    records[reference] = {"survivor_id": sid, "reference_id": reference,
                        "status": "VISIBLE", "first_seen_at": stamp, "last_seen_at": stamp,
                        "current_track_id": None, "track_history": [], "observations": [],
                        "latest_camera_confidence": track.confidence, "identity_score": o.score,
                        "estimated_map_position": None, "grid_cell": None, "position_uncertainty": None}
                    events.append({"event": "survivor_created", "survivor_id": sid, "reference_id": reference,
                                   "track_id": label, "sequence": sequence, "reason": "repeated_verified_appearance"})
            if reference:
                record = records[reference]
                sid = record["survivor_id"]
                is_new = reference not in self._records
                if not is_new and old_bindings.get(label) != reference:
                    duplicates += 1
                    events.append({"event": "duplicate_prevented", "survivor_id": sid, "reference_id": reference,
                                   "track_id": label, "sequence": sequence, "score": o.score, "margin": o.margin})
                bindings[label] = reference
                output[label] = sid
                history_key = f"{self.run_id}/{label}"
                if history_key not in record["track_history"]:
                    record["track_history"] = (record["track_history"] + [history_key])[-self.config.max_track_history:]
                record.update(status="VISIBLE", current_track_id=history_key, last_seen_at=stamp,
                              latest_camera_confidence=track.confidence, identity_score=o.score)
                if is_new or old_bindings.get(label) != reference or received_at - last_observation.get(reference, -float("inf")) >= self.config.observation_interval:
                    evidence = {"run_id": self.run_id, "track_id": label, "sequence": sequence,
                                "received_at_monotonic": received_at, "observed_at": stamp, "xyxy": list(track.xyxy),
                                "confidence": track.confidence, "identity_score": o.score, "margin": o.margin}
                    record["observations"] = (record["observations"] + [evidence])[-self.config.max_observations_per_survivor:]
                    last_observation[reference] = received_at
            else:
                pending_states[label] = reason
                if self._pending_states.get(label) != reason:
                    events.append({"event": "identity_pending", "track_id": label, "sequence": sequence, "reason": reason})
        # Weak visible tracks retain internal continuity, but are not shown/count-resolved.
        for label, reference in old_bindings.items():
            if label not in confirmed:
                bindings[label] = reference
        visible_refs = {bindings[label] for label in output}
        for reference, record in records.items():
            if reference not in visible_refs:
                if record["status"] == "VISIBLE":
                    events.append({"event": "survivor_not_visible", "survivor_id": record["survivor_id"], "sequence": sequence})
                record.update(status="NOT_VISIBLE", current_track_id=None)
        self._persist(records, duplicates, events)
        self._records, self._duplicates = records, duplicates
        self._bindings, self._candidates, self._pending_states = bindings, candidates, pending_states
        self._last_sequence, self._last_time = sequence, received_at
        self._last_observation = last_observation
        self._unavailable = False
        # Visible occupancy is a tracking fact; temporal/identity acceptance
        # are separate. Keep candidates visible while their identity is pending.
        return ManagerResult(len(tracks), len(output), len(tracks) - len(output), len(records), duplicates,
                             output, tuple(events), (time.perf_counter() - started) * 1000)

    def _release(self):
        if self._db is not None:
            self._db.close()
            self._db = None
        if self._lock is not None:
            self._lock.close()
            self._lock = None
            self._lock_path.unlink()  # Exact lock created by this instance, not mission data.
        self._closed = True

    def close(self):
        if self._closed:
            return
        try:
            if not self._failed:
                records = {key: {**value, "status": "NOT_VISIBLE", "current_track_id": None} for key, value in self._records.items()}
                self._persist(records, self._duplicates, [{"event": "mission_closed"}])
                self._records = records
        finally:
            self._release()


def annotate_survivors(image, verified, result):
    output = image.copy()
    for item in verified.tracks:
        sid = result.track_survivors.get(item.track.track_id)
        if sid:
            x, y, _, _ = item.track.xyxy
            cv2.putText(output, f"{sid} | estimated identity", (max(0, int(x)), max(50, min(output.shape[0] - 8, int(y) + 50))),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (80, 240, 240), 2, cv2.LINE_AA)
    return output
