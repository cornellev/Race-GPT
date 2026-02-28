# python write_telemetry.py blot.csv
# ollama create cev-efficiency-engineer -f Modelfile

import argparse
import json
from pathlib import Path

from telemetry import build_summary, load_telemetry_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        nargs="?",
        default="mock_baseline.csv",
        help="Telemetry CSV or JSON (default: mock_baseline.csv)",
    )
    parser.add_argument("--out", default="baseline_telemetry.json")
    args = parser.parse_args()

    df = load_telemetry_path(Path(args.input))
    summary = build_summary(df)

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Baseline written to {args.out}")


if __name__ == "__main__":
    main()
