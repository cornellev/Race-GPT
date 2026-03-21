import argparse
import csv
import json
import sqlite3
import struct
from pathlib import Path

CONFLICTING_FLAT_KEYS = {
    "seq",
    "power.ts",
    "power.current",
    "power.voltage",
}


def decode_std_msgs_string_cdr(blob: bytes) -> str:
    if len(blob) < 8:
        raise ValueError("ROS message blob too short to decode std_msgs/String.")

    string_length = struct.unpack_from("<I", blob, 4)[0]
    start = 8
    end = start + string_length
    if end > len(blob):
        raise ValueError("ROS message blob has invalid string length.")

    payload = blob[start:end]
    if payload.endswith(b"\x00"):
        payload = payload[:-1]
    return payload.decode("utf-8")


def flatten_dict(data: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, full_key))
        else:
            flat[full_key] = value
    return flat


def snapshot_to_csv_row(snapshot: dict, bag_timestamp_ns: int) -> dict:
    flat = flatten_dict(snapshot)
    power = snapshot.get("power") or {}

    row = {
        "bag_timestamp_ns": bag_timestamp_ns,
        "bag_timestamp_sec": bag_timestamp_ns / 1_000_000_000.0,
        "seq": snapshot.get("seq"),
        "global_ts": snapshot.get("global_ts"),
        "power_ts": power.get("ts"),
        "current": power.get("current"),
        "voltage": power.get("voltage"),
    }
    if row["current"] is not None and row["voltage"] is not None:
        row["power"] = row["current"] * row["voltage"]
    else:
        row["power"] = None

    row.update(
        {
            key: value
            for key, value in flat.items()
            if key not in CONFLICTING_FLAT_KEYS
        }
    )
    return row


def read_rosbag_snapshots(db3_path: Path) -> list[tuple[int, dict]]:
    query = """
        SELECT messages.timestamp, messages.data
        FROM messages
        JOIN topics ON topics.id = messages.topic_id
        WHERE topics.type = 'std_msgs/msg/String'
        ORDER BY messages.timestamp ASC
    """

    snapshots = []
    with sqlite3.connect(db3_path) as conn:
        for timestamp_ns, blob in conn.execute(query):
            message_text = decode_std_msgs_string_cdr(blob)
            payload = json.loads(message_text)
            if not isinstance(payload, dict):
                continue
            snapshots.append((timestamp_ns, payload))
    return snapshots


def write_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(snapshots: list[dict], output_path: Path, source_path: Path) -> None:
    payload = {
        "source_db3": str(source_path),
        "snapshots": snapshots,
    }
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2, allow_nan=True)


def resolve_bag_files(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix == ".db3":
        return [input_path]

    if (input_path / "metadata.yaml").exists():
        return sorted(input_path.glob("*.db3"))

    return sorted(input_path.glob("**/*.db3"))


def convert_bag(db3_path: Path, output_dir: Path, output_format: str) -> list[Path]:
    timestamped_snapshots = read_rosbag_snapshots(db3_path)
    if not timestamped_snapshots:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = db3_path.stem
    written_files = []

    snapshots = [snapshot for _, snapshot in timestamped_snapshots]
    rows = [snapshot_to_csv_row(snapshot, timestamp_ns) for timestamp_ns, snapshot in timestamped_snapshots]

    if output_format in {"csv", "both"}:
        csv_path = output_dir / f"{stem}.csv"
        write_csv(rows, csv_path)
        written_files.append(csv_path)

    if output_format in {"json", "both"}:
        json_path = output_dir / f"{stem}.json"
        write_json(snapshots, json_path, db3_path)
        written_files.append(json_path)

    return written_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ROS2 bag SQLite files containing std_msgs/String JSON snapshots into CSV or JSON."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="bags",
        help="Bag directory, single bag directory, or .db3 file (default: bags)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json", "both"],
        default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument(
        "--outdir",
        default="converted_bags",
        help="Directory for converted outputs (default: converted_bags)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    bag_files = resolve_bag_files(input_path)
    if not bag_files:
        raise SystemExit(f"No .db3 bag files found under {input_path}")

    output_dir = Path(args.outdir)
    for db3_path in bag_files:
        written_files = convert_bag(db3_path, output_dir, args.format)
        if not written_files:
            print(f"Skipped {db3_path}: no decodable JSON snapshots found.")
            continue
        for path in written_files:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
