import json
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = [
    "seq",
    "power_ts",
    "current",
    "voltage",
    "power",
]

COVARIANCE_FEATURE_COLUMNS = [
    "voltage",
    "current",
    "power",
]

CSV_RENAME_MAP = {
    "ros_time_sec": "power_ts",
    "bus_voltage": "voltage",
    "frame_id": "seq",
    "power.current": "current",
    "power.voltage": "voltage",
    "rpm_front.rpm_left": "rpm_front_left",
    "rpm_front.rpm_right": "rpm_front_right",
    "rpm_back.rpm_left": "rpm_back_left",
    "rpm_back.rpm_right": "rpm_back_right",
    "gps.lat": "lat",
    "gps.long": "long",
}


def safe_float(x):
    if pd.isna(x) or np.isinf(x):
        return 0.0
    return round(float(x), 10)


def stats(series: pd.Series):
    series = series.dropna()
    if len(series) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "var": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "p05": 0.0,
            "p95": 0.0,
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


def _flatten_snapshot(record: dict):
    row = {
        "seq": record.get("seq", 0),
        "power_ts": 0,
        "current": 0.0,
        "voltage": 0.0,
    }

    power = record.get("power") or {}
    row["power_ts"] = power.get("ts", row["power_ts"])
    row["current"] = power.get("current", row["current"])
    row["voltage"] = power.get("voltage", row["voltage"])
    return row


def _coerce_numeric(df: pd.DataFrame):
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    if aligned.iloc[:, 0].std() == 0 or aligned.iloc[:, 1].std() == 0:
        return 0.0
    return safe_float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def _theil_sen_fit(x: np.ndarray, y: np.ndarray):
    if len(x) < 2:
        return 0.0, safe_float(np.median(y)) if len(y) else 0.0

    slopes = []
    for i in range(len(x) - 1):
        dx = x[i + 1:] - x[i]
        valid = np.abs(dx) > 1e-9
        if not np.any(valid):
            continue
        dy = y[i + 1:] - y[i]
        slopes.extend((dy[valid] / dx[valid]).tolist())

    if not slopes:
        slope = 0.0
    else:
        slope = float(np.median(slopes))

    intercept = float(np.median(y - slope * x))
    return safe_float(slope), safe_float(intercept)


def electrical_dynamics(df: pd.DataFrame):
    if len(df) < 3:
        return {
            "delta_count": 0,
            "dv_di_slope": 0.0,
            "dv_di_intercept": 0.0,
            "residual_mad": 0.0,
            "outlier_fraction": 0.0,
            "inst_slope_p05": 0.0,
            "inst_slope_p95": 0.0,
        }

    d_i = df.current.diff().iloc[1:]
    d_v = df.voltage.diff().iloc[1:]
    valid = d_i.abs() > 1e-9
    d_i = d_i[valid].to_numpy()
    d_v = d_v[valid].to_numpy()

    if len(d_i) < 2:
        return {
            "delta_count": int(len(d_i)),
            "dv_di_slope": 0.0,
            "dv_di_intercept": 0.0,
            "residual_mad": 0.0,
            "outlier_fraction": 0.0,
            "inst_slope_p05": 0.0,
            "inst_slope_p95": 0.0,
        }

    slope, intercept = _theil_sen_fit(d_i, d_v)
    fitted = slope * d_i + intercept
    residual = d_v - fitted
    residual_abs = np.abs(residual)
    mad = float(np.median(residual_abs))
    outlier_threshold = max(1e-6, 3.0 * mad)
    outlier_fraction = float((residual_abs > outlier_threshold).mean())

    inst_slope = d_v / d_i

    return {
        "delta_count": int(len(d_i)),
        "dv_di_slope": safe_float(slope),
        "dv_di_intercept": safe_float(intercept),
        "residual_mad": safe_float(mad),
        "outlier_fraction": safe_float(outlier_fraction),
        "inst_slope_p05": safe_float(np.quantile(inst_slope, 0.05)),
        "inst_slope_p95": safe_float(np.quantile(inst_slope, 0.95)),
    }


def normalize_telemetry_df(df: pd.DataFrame):
    df = df.dropna(how="all").copy()

    if {"power", "driver", "rpm_front", "rpm_back", "gps"}.issubset(df.columns):
        rows = [_flatten_snapshot(rec) for rec in df.to_dict(orient="records")]
        df = pd.DataFrame(rows)
    else:
        df = df.rename(columns=CSV_RENAME_MAP)
        if "power" not in df.columns and {"voltage", "current"}.issubset(df.columns):
            df["power"] = pd.to_numeric(df["voltage"], errors="coerce") * pd.to_numeric(
                df["current"], errors="coerce"
            )

    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    df = _coerce_numeric(df[CANONICAL_COLUMNS])
    df["power"] = df["voltage"] * df["current"]
    return df.fillna(0.0)


def load_telemetry_path(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".json":
        with open(path) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            if "snapshots" in raw and isinstance(raw["snapshots"], list):
                df = pd.DataFrame(raw["snapshots"])
            else:
                df = pd.DataFrame([raw])
        else:
            df = pd.DataFrame(raw)
    else:
        raise ValueError("Unsupported file type (use .csv or .json)")

    return normalize_telemetry_df(df)


def load_telemetry_csv_string(csv_text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(csv_text))
    return normalize_telemetry_df(df)


def load_telemetry_json_obj(obj) -> pd.DataFrame:
    if isinstance(obj, str):
        obj = json.loads(obj)
    if isinstance(obj, dict):
        if "snapshots" in obj and isinstance(obj["snapshots"], list):
            df = pd.DataFrame(obj["snapshots"])
        else:
            df = pd.DataFrame([obj])
    elif isinstance(obj, list):
        df = pd.DataFrame(obj)
    else:
        raise ValueError("Unsupported JSON telemetry payload.")
    return normalize_telemetry_df(df)


def covariance_matrix(df: pd.DataFrame):
    candidate_cols = [c for c in COVARIANCE_FEATURE_COLUMNS if c in df.columns]
    numeric_df = df[candidate_cols].select_dtypes(include=[np.number])
    if numeric_df.empty:
        return {}

    cov_df = numeric_df.cov().fillna(0.0)
    return {
        row: {
            col: safe_float(cov_df.loc[row, col])
            for col in cov_df.columns
            if col != row
        }
        for row in cov_df.index
    }


def build_summary(df: pd.DataFrame):
    corr = {
        "voltage_vs_current": safe_corr(df.voltage, df.current),
        "current_vs_power": safe_corr(df.current, df.power),
    }
    dynamics = electrical_dynamics(df)

    return {
        "meta": {
            "total_rows": int(len(df)),
            "seq_start": safe_float(df.seq.min()),
            "seq_end": safe_float(df.seq.max()),
            "time_start_sec": safe_float(df.power_ts.min()),
            "time_end_sec": safe_float(df.power_ts.max()),
        },
        "voltage": {
            "overall": stats(df.voltage),
            "min_sag": safe_float(df.voltage.min()),
        },
        "current": {
            "overall": stats(df.current),
            "idle_fraction": safe_float(
                (df.current.abs() < 0.01).mean()
            ),
            "peak": safe_float(df.current.max()),
        },
        "power": {
            "overall": stats(df.power),
            "peak": safe_float(df.power.max()),
            "negative_fraction": safe_float((df.power < 0).mean()),
        },
        "efficiency_signals": {
            "power_per_amp_mean": safe_float(
                df.power.mean() / df.current.mean()
                if df.current.mean() != 0 else 0.0
            ),
            "power_std": safe_float(df.power.std()),
        },
        "dynamics": dynamics,
        "correlations": corr,
        "covariance": covariance_matrix(df),
    }
