import asyncio
import json
import os
import traceback
import multiprocessing as mp
from typing import Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ollama import chat
from pydantic import BaseModel, Field

from telemetry import (
    build_summary,
    load_telemetry_csv_string,
    load_telemetry_json_obj,
)

app = FastAPI()
analysis_lock = asyncio.Lock()
ANALYSIS_TIMEOUT_SEC = float(os.getenv("ANALYSIS_TIMEOUT_SEC", "15"))


class TelemetryIn(BaseModel):
    csv: Optional[str] = None
    json_payload: Optional[Any] = Field(default=None, alias="json")


BASELINE_PATH = "baseline_telemetry.json"


def load_baseline():
    with open(BASELINE_PATH) as f:
        return json.load(f)


def precheck(baseline: dict, current_summary: dict, current_df):
    if len(current_df) < 5:
        return "Telemetry invalid: insufficient rows."
    if "power_ts" in current_df.columns:
        if current_df.power_ts.nunique() <= 1:
            return "Telemetry invalid: time not increasing."

    b = baseline
    c = current_summary

    base_voltage = b.get("voltage") or b.get("bus_voltage") or {}
    current_voltage = c.get("voltage") or c.get("bus_voltage") or {}
    base_v_p05 = base_voltage.get("overall", {}).get("p05", 0.0)
    current_v_min = current_voltage.get("min_sag")
    if current_v_min is None:
        current_v_min = current_voltage.get("overall", {}).get("min", 0.0)
    base_current_mean = b.get("current", {}).get("overall", {}).get("mean", 0.0)
    current_current_peak = c.get("current", {}).get("peak")
    if current_current_peak is None:
        current_current_peak = c.get("current", {}).get("overall", {}).get(
            "max", 0.0
        )
    base_dyn = b.get("dynamics", {})
    base_slope = base_dyn.get("dv_di_slope", 0.0)
    load_delta_i = max(0.0, current_current_peak - base_current_mean)
    load_adjust = min(0.0, base_slope * load_delta_i)
    load_adjust = max(load_adjust, -1.5)
    sag_floor = (base_v_p05 - 1.0) + load_adjust
    if current_v_min < sag_floor:
        return "Critical: voltage sag below baseline."

    base_current_p95 = b.get("current", {}).get("overall", {}).get("p95", 0.0)
    current_current_max = c.get("current", {}).get("peak")
    if current_current_max is None:
        current_current_max = c.get("current", {}).get("overall", {}).get(
            "max", 0.0
        )
    if base_current_p95 > 0 and current_current_max > (base_current_p95 * 1.5):
        return "Warning: current spike above baseline."

    mean_power = c.get("power", {}).get("overall", {}).get("mean", 0.0)
    mean_bus = current_voltage.get("overall", {}).get("mean", 0.0)
    mean_current = c.get("current", {}).get("overall", {}).get("mean", 0.0)
    estimated_power = mean_bus * mean_current
    denom = max(1.0, abs(mean_power))
    if abs(mean_power - estimated_power) / denom > 0.3:
        return "Warning: power/current mismatch detected."

    base_dyn = b.get("dynamics", {})
    curr_dyn = c.get("dynamics", {})
    min_delta_count = 6
    if (
        base_dyn.get("delta_count", 0) >= min_delta_count
        and curr_dyn.get("delta_count", 0) >= min_delta_count
    ):
        base_p05 = base_dyn.get("inst_slope_p05", 0.0)
        base_p95 = base_dyn.get("inst_slope_p95", 0.0)
        base_mad = base_dyn.get("residual_mad", 0.0)
        curr_slope = curr_dyn.get("dv_di_slope", 0.0)
        curr_outlier_fraction = curr_dyn.get("outlier_fraction", 0.0)
        base_outlier_fraction = base_dyn.get("outlier_fraction", 0.0)

        slope_margin = max(0.05, 3.0 * base_mad)
        slope_low = base_p05 - slope_margin
        slope_high = base_p95 + slope_margin
        outlier_limit = min(0.95, base_outlier_fraction + 0.35)

        if (curr_slope < slope_low or curr_slope > slope_high) and (
            curr_outlier_fraction > outlier_limit
        ):
            return "Warning: voltage/current dynamics deviate from baseline."

    return None


def parse_incoming_payload(csv_payload: Optional[str], json_payload: Optional[Any]):
    if csv_payload:
        return load_telemetry_csv_string(csv_payload)
    if json_payload is not None:
        return load_telemetry_json_obj(json_payload)
    raise ValueError("Missing telemetry payload: provide `csv` or `json`.")


def analyze_payload(csv_payload: Optional[str], json_payload: Optional[Any]):
    try:
        try:
            current_df = parse_incoming_payload(csv_payload, json_payload)
        except Exception:
            return {"verdict": "Telemetry invalid: unable to parse payload."}

        baseline = load_baseline()
        current_summary = build_summary(current_df)
        decision = precheck(baseline, current_summary, current_df)

        if decision is None:
            return {"verdict": "No critical issues detected."}

        prompt = f"""
        PRECHECK_DECISION:
        {decision}

        BASELINE_TELEMETRY:
        {json.dumps(baseline, separators=(",", ":"), ensure_ascii=False)}

        CURRENT_SUMMARY:
        {json.dumps(current_summary, separators=(",", ":"), ensure_ascii=False)}

        Return exactly one sentence describing the issue. Do not add extra details beyond the decision.
        """.strip()

        print(prompt, flush=True)

        try:
            response = chat(
                model="cev-efficiency-engineer",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                keep_alive=10000,
                options={
                    "stop": ["\n"],
                },
            )
        except Exception:
            traceback.print_exc()
            return {"verdict": "Model unavailable: unable to analyze."}

        text = response.get("message", {}).get("content", "")
        if not isinstance(text, str) or not text.strip():
            return {"verdict": decision}

        text = text.strip()
        if "." in text:
            text = text.split(".")[0].strip() + "."
        else:
            text = text.strip() + "."

        return {"verdict": text}
    except Exception:
        traceback.print_exc()
        return {"verdict": "Internal analysis error: unable to analyze."}


async def run_analysis(csv_payload: Optional[str], json_payload: Optional[Any]):
    if analysis_lock.locked():
        return {"verdict": "LLM reasoning already in progress. Try again shortly."}

    try:
        async with analysis_lock:
            return await asyncio.to_thread(run_analysis_sync, csv_payload, json_payload)
    except Exception:
        traceback.print_exc()
        return {"verdict": "Internal analysis error: unable to analyze."}




def _analysis_process_entry(result_queue: mp.Queue, csv_payload: Optional[str], json_payload: Optional[Any]):
    try:
        result_queue.put(analyze_payload(csv_payload, json_payload))
    except Exception:
        traceback.print_exc()
        result_queue.put({"verdict": "Internal analysis error: unable to analyze."})


def run_analysis_sync(csv_payload: Optional[str], json_payload: Optional[Any]):
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    process = mp.Process(
        target=_analysis_process_entry,
        args=(result_queue, csv_payload, json_payload),
        daemon=True,
    )
    process.start()
    process.join(ANALYSIS_TIMEOUT_SEC)

    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)
        return {"verdict": "Model timed out: unable to analyze."}

    try:
        return result_queue.get_nowait()
    except Exception:
        return {"verdict": "Internal analysis error: unable to analyze."}


@app.post("/analyze")
async def analyze(data: TelemetryIn):
    try:
        return await run_analysis(data.csv, data.json_payload)
    except Exception:
        traceback.print_exc()
        return {"verdict": "Internal analysis error: unable to analyze."}


@app.websocket("/ws/analyze")
async def analyze_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                payload = await websocket.receive_json()
                result = await run_analysis(
                    payload.get("csv"),
                    payload.get("json"),
                )
                await websocket.send_json(result)
            except WebSocketDisconnect:
                return
            except Exception:
                traceback.print_exc()
                await websocket.send_json({"verdict": "Internal analysis error: unable to analyze."})
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
