#!/usr/bin/env python3
"""Reload an observed daily-flow model and recursively predict from a CSV.

The CSV must identify its station, observed-input status and m³/s units. The
forecast origin is the final supplied observation date, never the system date.
New observations are not fetched and unknown future labels are left blank.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.observed_rivers import sha256, validate_records
from src.models.observed_rivers import load_checkpoint, recursive_forecast

STATIONS = {"yellow_river": "lijin", "mississippi": "USGS-07010000"}


def read_input(path, station_id, lookback):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"date", "station_id", "discharge", "unit", "source_status"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Input needs columns: " + ", ".join(sorted(required)))
        records = []
        for row in reader:
            if row["station_id"] != station_id:
                raise ValueError("Input station differs from the checkpoint target")
            if row["source_status"] != "observed":
                raise ValueError("Explicit observed input status required; no synthetic fallback")
            if row["unit"] not in ("m³/s", "m3/s"):
                raise ValueError("Input discharge must already be in m³/s")
            records.append({"date": date.fromisoformat(row["date"]).isoformat(), "discharge": float(row["discharge"])})
    records = validate_records(records)
    if len(records) < lookback:
        raise ValueError("Not enough observed history for the checkpoint lookback")
    history = records[-lookback:]
    beginning = date.fromisoformat(history[0]["date"])
    if any((date.fromisoformat(record["date"])-beginning).days != index for index,record in enumerate(history)):
        raise ValueError("The last history window has missing days; interpolation is not performed")
    return records, np.asarray([record["discharge"] for record in history], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 30:
        raise ValueError("Choose 1–30 recursive prediction days")
    if args.output.resolve() in (args.input.resolve(), args.checkpoint.resolve()):
        raise ValueError("Forecast output must not overwrite input data or checkpoint")
    torch.set_num_threads(1)
    model, scaler, checkpoint = load_checkpoint(args.checkpoint)
    target = checkpoint["target"]
    if target not in STATIONS:
        raise ValueError("Unknown target station contract")
    lookback = int(checkpoint["protocol"]["lookback_days"])
    records, history = read_input(args.input, STATIONS[target], lookback)
    origin = records[-1]["date"]
    predicted = recursive_forecast(model, history, scaler, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["origin_date", "date", "lead_days", "station_id", "unit", "prediction", "observed", "prediction_status", "input_source_status"], lineterminator="\n")
        writer.writeheader()
        for index,value in enumerate(predicted):
            writer.writerow({"origin_date": origin, "date": (date.fromisoformat(origin)+timedelta(days=index+1)).isoformat(),
                             "lead_days": index+1, "station_id": STATIONS[target], "unit": "m³/s", "prediction": float(value),
                             "observed": "", "prediction_status": "model_prediction", "input_source_status": "observed"})
    metadata = {"protocol_version": checkpoint["protocol_version"], "target": target, "station_id": STATIONS[target],
                "seed": checkpoint["seed"], "method": checkpoint["method"], "lookback_days": lookback,
                "origin_date": origin, "forecast_steps": args.steps, "forecast_mode": "recursive_from_supplied_observed_cutoff",
                "current_data_fetched": False, "future_observed_labels_supplied": False, "input_sha256": sha256(args.input),
                "checkpoint_sha256": sha256(args.checkpoint), "training_source_hashes": checkpoint["source_hashes"],
                "note": "Predictions start after the final supplied observation date. The CSV's observed status is an explicit input declaration, not independent verification of newly supplied measurements."}
    metadata_path = args.output.with_name(args.output.stem+"_metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Wrote {args.steps} recursive daily predictions from {origin}; unknown future observations remain blank.")


if __name__ == "__main__":
    main()
