import pandas as pd
import numpy as np
import json
from ollama import chat

df = pd.read_csv("blot.csv")
df = df.dropna(how="all")

RECENT_N = 25
recent = df.tail(RECENT_N)

def safe_float(x):
    """Convert NaN/inf to 0.0 so JSON + model stay sane."""
    if pd.isna(x) or np.isinf(x):
        return 0.0
    return float(x)

def stats(series):
    series = series.dropna()
    if len(series) == 0:
        return {
            "mean": 0.0, "std": 0.0, "var": 0.0,
            "min": 0.0, "max": 0.0,
            "median": 0.0, "p05": 0.0, "p95": 0.0,
            "cv": 0.0,
        }

    mean = series.mean()
    std = series.std()

    return {
        "mean": safe_float(mean),
        "std": safe_float(std),
        "var": safe_float(series.var()),
        "min": safe_float(series.min()),
        "max": safe_float(series.max()),
        "median": safe_float(series.median()),
        "p05": safe_float(series.quantile(0.05)),
        "p95": safe_float(series.quantile(0.95)),
        "cv": safe_float(std / mean) if mean != 0 else 0.0,
    }

summary = {
    "meta": {
        "total_rows": int(len(df)),
        "recent_rows": int(len(recent)),
        "time_start_sec": safe_float(df.ros_time_sec.min()),
        "time_end_sec": safe_float(df.ros_time_sec.max()),
    },

    "bus_voltage": {
        "overall": stats(df.bus_voltage),
        "recent": stats(recent.bus_voltage),
        "delta_recent_vs_overall": safe_float(
            recent.bus_voltage.mean() - df.bus_voltage.mean()
        ),
        "recent_min_sag": safe_float(recent.bus_voltage.min()),
    },

    "shunt_voltage": {
        "overall": stats(df.shunt_voltage),
        "recent": stats(recent.shunt_voltage),
        "noise_ratio_recent": safe_float(
            recent.shunt_voltage.std() / (df.shunt_voltage.std() + 1e-6)
        ),
    },

    "current": {
        "overall": stats(df.current),
        "recent": stats(recent.current),
        "recent_idle_fraction": safe_float(
            (recent.current.abs() < 0.01).mean()
        ),
        "recent_peak": safe_float(recent.current.max()),
        "delta_recent_vs_overall": safe_float(
            recent.current.mean() - df.current.mean()
        ),
    },

    "power": {
        "overall": stats(df.power),
        "recent": stats(recent.power),
        "recent_peak": safe_float(recent.power.max()),
        "recent_negative_fraction": safe_float(
            (recent.power < 0).mean()
        ),
        "delta_recent_vs_overall": safe_float(
            recent.power.mean() - df.power.mean()
        ),
    },

    "efficiency_signals": {
        "power_per_amp_mean": safe_float(
            df.power.mean() / df.current.mean()
            if df.current.mean() != 0 else 0.0
        ),
        "recent_power_per_amp_mean": safe_float(
            recent.power.mean() / recent.current.mean()
            if recent.current.mean() != 0 else 0.0
        ),
        "power_variability_ratio": safe_float(
            recent.power.std() / (df.power.std() + 1e-6)
        ),
    },
}

telemetry_payload = json.dumps(
    summary,
    separators=(",", ":"),
    ensure_ascii=False,
)

telemetry_payload = telemetry_payload.encode("utf-8", "ignore").decode()


CURRENT_K = 8
RECENT_WINDOW_ROWS = 8  # or however far back you want to sample from

current_df = pd.read_csv("bad.csv")
current_df = current_df.dropna(how="all")

recent_slice = current_df.tail(RECENT_WINDOW_ROWS)

if len(recent_slice) <= CURRENT_K:
    sampled = recent_slice
else:
    sampled = recent_slice.sample(
        n=CURRENT_K,
        random_state=None  # truly random each run
    )

current_telemetry_csv = sampled.to_csv(index=False)



# ---- OLLAMA CALL ----

final_prompt = f"""
TELEMETRY_EXPECTED:
{telemetry_payload}

CURRENT_TELEMETRY:
{current_telemetry_csv}
""".strip()


stream = chat(
    model="cev-efficiency-engineer",
    messages=[
        {"role": "user", "content": final_prompt},
    ],
    stream=True,
    keep_alive=600,
)

output = ""

for chunk in stream:
    msg = chunk.get("message")
    if not msg:
        continue

    token = msg.get("content")
    if not token:
        continue

    output += token
    print(token, end="", flush=True)

    # HARD STOP CONDITIONS
    if "\n" in output:
        break
    if output.count(".") >= 1:
        break

print()
