"""Read-only mission summary/export; never includes appearance vectors or pixels."""
import argparse
import json
from pathlib import Path
import sqlite3


def read_report(path: Path, event_limit: int = 100) -> dict:
    if type(event_limit) is not int or not 0 <= event_limit <= 1000:
        raise ValueError("Event report limit must be 0..1000.")
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
    try:
        db.execute("BEGIN")  # Consistent read snapshot even if the live writer commits.
        metadata = dict(db.execute("SELECT key, value FROM metadata WHERE key != 'gallery'"))
        if metadata.get("schema") != "1":
            raise ValueError("Unsupported mission report schema.")
        records = [json.loads(row[0]) for row in db.execute("SELECT record FROM survivors")]
        records.sort(key=lambda record: int(record["survivor_id"][1:]))
        events = [json.loads(row[0]) for row in db.execute("SELECT payload FROM events ORDER BY id DESC LIMIT ?", (event_limit,))]
        return {"mission_id": metadata["mission_id"], "unique_survivors": len(records),
                "duplicates_prevented": int(metadata["duplicates_prevented"]), "current_persons": None,
                "meaning": "archival appearance-based estimates, not live occupancy or medical status",
                "records": records, "recent_events": list(reversed(events))}
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--events", type=int, default=100)
    args = parser.parse_args()
    report = read_report(args.database, args.events)
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as out:
            out.write(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
