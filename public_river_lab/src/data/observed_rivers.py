"""Strict, read-only imports of the two attributed daily-discharge archives.

Original XLSX/GeoJSON files remain untouched. No substitute synthetic records,
interpolation or model-derived discharge are accepted as station observations.
"""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np

CFS_TO_CMS = 0.028316846592
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_records(records):
    seen = set()
    for record in records:
        day = date.fromisoformat(record["date"])
        if day in seen:
            raise ValueError("Duplicate source calendar date: " + day.isoformat())
        seen.add(day)
        if not np.isfinite(record["discharge"]) or record["discharge"] < 0:
            raise ValueError("Invalid observed discharge on " + day.isoformat())
    if not records:
        raise ValueError("No observed discharge records")
    return sorted(records, key=lambda record: record["date"])


def read_lijin(path):
    """Read the published workbook's numeric XML without spreadsheet edits."""
    with zipfile.ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        properties = workbook.find("s:workbookPr", NS)
        if properties is not None and properties.get("date1904") in ("1", "true"):
            raise ValueError("Unexpected Lijin Excel date system")
        shared = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = ["".join(item.itertext()) for item in shared]
        rows = ET.fromstring(archive.read("xl/worksheets/sheet1.xml")).findall("s:sheetData/s:row", NS)
        headers = {}
        for cell in rows[0].findall("s:c", NS):
            value = cell.find("s:v", NS)
            if value is not None:
                headers[cell.get("r")] = strings[int(value.text)] if cell.get("t") == "s" else value.text
        if headers.get("A1") != "Time (year/month/day)" or headers.get("B1") != "Discharge (m3/s)":
            raise ValueError("Unknown Lijin date/discharge headers or units")
        records = []
        for row in rows[1:]:
            values = {}
            for cell in row.findall("s:c", NS):
                value = cell.find("s:v", NS)
                if value is not None:
                    values["".join(character for character in cell.get("r") if character.isalpha())] = value.text
            if "A" not in values or "B" not in values:
                continue
            serial = float(values["A"])
            if not serial.is_integer():
                raise ValueError("Lijin source includes a sub-daily timestamp")
            day = (date(1899, 12, 30) + timedelta(days=int(serial))).isoformat()
            records.append({"date": day, "discharge": float(values["B"]), "source_status": "observed", "qualifier": "", "approval_status": "not supplied"})
    return validate_records(records)


def read_usgs(path):
    response = json.loads(Path(path).read_text(encoding="utf-8"))
    if any(link.get("rel") == "next" for link in response.get("links", [])):
        raise ValueError("Incomplete paginated USGS response")
    records = []
    for feature in response["features"]:
        props = feature["properties"]
        if (props.get("monitoring_location_id"), props.get("parameter_code"), props.get("statistic_id"), props.get("unit_of_measure")) != ("USGS-07010000", "00060", "00003", "ft^3/s"):
            raise ValueError("Unexpected USGS station, discharge statistic or units")
        if props.get("value") in (None, ""):
            continue
        flags = props.get("qualifier") or []
        if isinstance(flags, str):
            flags = [flags]
        records.append({"date": date.fromisoformat(props["time"]).isoformat(), "discharge": float(props["value"]) * CFS_TO_CMS,
                        "source_status": "observed", "qualifier": ";".join(flags), "approval_status": props.get("approval_status", "")})
    return validate_records(records)


def load_sources(root):
    root = Path(root).resolve()
    provenance = json.loads((root / "experiments/observed_rivers/provenance.json").read_text(encoding="utf-8"))
    readers = {"yellow_river": read_lijin, "mississippi": read_usgs}
    if {source["river_id"] for source in provenance["sources"]} != set(readers):
        raise ValueError("This experiment requires exactly the attributed Yellow River and Mississippi sources")
    period = provenance["common_period"]
    observations = {}
    for source in provenance["sources"]:
        path = (root / source["raw_file"]).resolve()
        if not path.is_relative_to(root / "experiments/observed_rivers/raw"):
            raise ValueError("Raw source outside the explicit source archive directory")
        if sha256(path) != source["raw_sha256"]:
            raise ValueError("Raw source checksum differs from attributed archive: " + source["river_id"])
        if not source.get("redistribution_allowed"):
            raise ValueError("Source has no explicit publication permission")
        all_records = readers[source["river_id"]](path)
        records = [record for record in all_records if period["start"] <= record["date"] <= period["end"]]
        if len(all_records) != source["raw_observations"] or len(records) != source["observations_in_common_period"]:
            raise ValueError("Source row counts differ from attributed archive")
        observations[source["river_id"]] = records
    return observations, provenance


def daily_windows(records, period, lookback=30):
    """Calendar partitions with consecutive causal histories; gaps are skipped."""
    records = validate_records(records)
    start, end = date.fromisoformat(period["start"]), date.fromisoformat(period["end"])
    if lookback < 1 or end <= start:
        raise ValueError("Invalid study period or history length")
    calendar = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    train_end, validation_end = int(len(calendar) * .7), int(len(calendar) * .85)
    if train_end <= lookback or validation_end <= train_end or validation_end >= len(calendar):
        raise ValueError("Insufficient study history")
    available = {date.fromisoformat(record["date"]): float(record["discharge"]) for record in records}
    parts = {}
    for name, lower, upper in [("train", 0, train_end), ("validation", train_end, validation_end), ("test", validation_end, len(calendar))]:
        inputs, targets, dates, skipped = [], [], [], 0
        for index in range(max(lookback, lower), upper):
            required = calendar[index-lookback:index+1]
            if any(day not in available for day in required):
                skipped += 1
                continue
            inputs.append([[available[day]] for day in required[:-1]])
            targets.append([available[required[-1]]])
            dates.append(required[-1].isoformat())
        if not inputs:
            raise ValueError("No complete consecutive observed windows in " + name)
        parts[name] = {"x": np.asarray(inputs, dtype=np.float32), "y": np.asarray(targets, dtype=np.float64), "dates": dates,
                       "start": calendar[lower].isoformat(), "end": calendar[upper-1].isoformat(), "skipped_windows": skipped}
    raw_train = np.asarray([available[day] for day in calendar[:train_end] if day in available], dtype=np.float64)
    return parts, raw_train
