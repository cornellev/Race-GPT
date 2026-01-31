# python write_telemetry.py blot.csv --recent-n 25
# ollama create cev-efficiency-engineer -f Modelfile

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path


def safe_float(x):
    if pd.isna(x) or np.isinf(x):
        return 0.0
    return float(x)


def stats(series: pd.Series):
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


def load_telemetry(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".json":
        with open(path) as f:
            raw = json.load(f)
        df = pd.DataFrame(raw)
    else:
        raise ValueError("Unsupported file type (use .csv or .json)")

    return df.dropna(how="all")


def build_summary(df: pd.DataFrame, recent_n: int):
    recent = df.tail(recent_n)

    return {
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Telemetry CSV or JSON")
    parser.add_argument("--recent-n", type=int, default=25)
    parser.add_argument("--out", default="baseline_telemetry.json")
    args = parser.parse_args()

    df = load_telemetry(Path(args.input))
    summary = build_summary(df, args.recent_n)

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Baseline written to {args.out}")


if __name__ == "__main__":
    main()
