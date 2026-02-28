import asyncio
import csv
import io
import json
import time

import websockets

WS_URL = "ws://localhost:8000/ws/analyze"

CSV_COLUMNS = [
    "seq",
    "power_ts",
    "current",
    "voltage",
    "driver_ts",
    "throttle",
    "velocity",
    "turn_angle",
    "rpm_front_ts",
    "rpm_front_left",
    "rpm_front_right",
    "rpm_back_ts",
    "rpm_back_left",
    "rpm_back_right",
    "gps_ts",
    "lat",
    "long",
]


def rows_to_csv(rows):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return out.getvalue()


def rows_to_snapshot_json(rows):
    snapshots = []
    for row in rows:
        snapshots.append(
            {
                "seq": int(row["seq"]),
                "power": {
                    "ts": int(row["power_ts"]),
                    "current": float(row["current"]),
                    "voltage": float(row["voltage"]),
                },
                "driver": {
                    "ts": int(row["driver_ts"]),
                    "throttle": float(row["throttle"]),
                    "velocity": float(row["velocity"]),
                    "turn_angle": float(row["turn_angle"]),
                },
                "rpm_front": {
                    "ts": int(row["rpm_front_ts"]),
                    "rpm_left": float(row["rpm_front_left"]),
                    "rpm_right": float(row["rpm_front_right"]),
                },
                "rpm_back": {
                    "ts": int(row["rpm_back_ts"]),
                    "rpm_left": float(row["rpm_back_left"]),
                    "rpm_right": float(row["rpm_back_right"]),
                },
                "gps": {
                    "ts": int(row["gps_ts"]),
                    "lat": float(row["lat"]),
                    "long": float(row["long"]),
                },
            }
        )
    return snapshots


good_rows = [
    {
        "seq": i,
        "power_ts": i,
        "current": 0.25 + 0.01 * (i % 3),
        "voltage": 40.2 - 0.03 * (i % 3),
        "driver_ts": i,
        "throttle": 0.35 + 0.01 * (i % 4),
        "velocity": 8.2 + 0.12 * i,
        "turn_angle": 0.03 + 0.002 * (i % 5),
        "rpm_front_ts": i,
        "rpm_front_left": 520 + 6 * i,
        "rpm_front_right": 522 + 6 * i,
        "rpm_back_ts": i,
        "rpm_back_left": 519 + 6 * i,
        "rpm_back_right": 521 + 6 * i,
        "gps_ts": i,
        "lat": 42.445 + 0.00001 * i,
        "long": -76.483 - 0.00001 * i,
    }
    for i in range(1, 13)
]

bad_rows = [
    {
        "seq": i,
        "power_ts": i,
        "current": 0.30 + 0.07 * i,
        "voltage": 39.6 - 0.25 * i,
        "driver_ts": i,
        "throttle": 0.55 + 0.02 * i,
        "velocity": 9.0 + 0.1 * i,
        "turn_angle": 0.06 + 0.003 * i,
        "rpm_front_ts": i,
        "rpm_front_left": 600 + 12 * i,
        "rpm_front_right": 590 + 12 * i,
        "rpm_back_ts": i,
        "rpm_back_left": 598 + 12 * i,
        "rpm_back_right": 588 + 12 * i,
        "gps_ts": i,
        "lat": 42.445 + 0.00002 * i,
        "long": -76.483 - 0.00002 * i,
    }
    for i in range(1, 9)
]

# Expected-normal pattern: increasing current with mild voltage drop (inverse relation).
correlation_rows = [
    {
        "seq": i,
        "power_ts": i,
        "current": 0.15 + 0.03 * i,
        "voltage": 40.9 - 0.18 * i,
        "driver_ts": i,
        "throttle": 0.40 + 0.01 * i,
        "velocity": 8.8 + 0.15 * i,
        "turn_angle": 0.04 + 0.001 * i,
        "rpm_front_ts": i,
        "rpm_front_left": 560 + 9 * i,
        "rpm_front_right": 559 + 9 * i,
        "rpm_back_ts": i,
        "rpm_back_left": 558 + 9 * i,
        "rpm_back_right": 557 + 9 * i,
        "gps_ts": i,
        "lat": 42.445 + 0.000015 * i,
        "long": -76.483 - 0.000015 * i,
    }
    for i in range(1, 9)
]


async def run():
    scenarios = [
        #("good_csv", {"csv": rows_to_csv(good_rows)}),
        ("good_json", {"json": rows_to_snapshot_json(good_rows)}),
        #("bad_csv", {"csv": rows_to_csv(bad_rows)}),
        ("bad_json", {"json": rows_to_snapshot_json(bad_rows)}),
        #("correlation_csv", {"csv": rows_to_csv(correlation_rows)}),
        ("correlation_json", {"json": rows_to_snapshot_json(correlation_rows)}),
    ]

    async with websockets.connect(WS_URL) as ws:
        for label, payload in scenarios:
            start = time.perf_counter()
            await ws.send(json.dumps(payload))
            response = await ws.recv()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            print(f"{label} ({elapsed_ms:.1f} ms)", json.loads(response))


if __name__ == "__main__":
    asyncio.run(run())
