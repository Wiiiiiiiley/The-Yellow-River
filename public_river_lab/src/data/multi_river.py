"""Reproducible educational river scenarios and strict hourly observation import.

Scenario widths, slopes, roughness, discharge and concentration are assumptions,
not measured geometry or calibrated basin parameters. Five generic reaches form
an illustrative directed chain. The rainfall/runoff and sediment relations are
causal scenario generators, not operational hydrological forecasts.
"""

from __future__ import annotations

import calendar
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

FEATURE_NAMES = ("water_depth", "discharge", "sediment", "temperature", "rainfall", "season_sin")
FEATURE_UNITS = ("m", "m³/s", "kg/m³", "°C", "mm/h", "1")
TARGET_NAMES = FEATURE_NAMES[:3]
START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def river_seed(river_id: str, seed: int = 42) -> int:
    """Stable across Python processes, unlike Python's randomized hash()."""
    digest = hashlib.sha256(f"{seed}:{river_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def load_profiles(path: Path) -> dict[str, dict]:
    """Load and validate the public scenario registry."""
    profiles = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("River profiles must be a nonempty object")
    required = {"id", "name_zh", "name_en", "region", "summary", "hydrology", "width_m", "slope", "manning_n", "base_discharge", "sediment_scale", "flood_peak_day", "lag_hours", "reach_names", "source_ids", "source_status", "parameter_status"}
    for key, profile in profiles.items():
        if not isinstance(profile, dict) or required - profile.keys():
            raise ValueError(f"Incomplete river profile: {key}")
        if profile["id"] != key or len(profile["reach_names"]) != 5:
            raise ValueError(f"Profile {key} must identify five generic reaches")
        ids = profile.get("station_ids", [f"r{i + 1}" for i in range(5)])
        if len(ids) != 5 or len(set(ids)) != 5:
            raise ValueError(f"Profile {key} requires five distinct reach IDs")
        for field in ("width_m", "slope", "manning_n", "base_discharge", "sediment_scale", "lag_hours"):
            if not np.isfinite(profile[field]) or float(profile[field]) <= 0:
                raise ValueError(f"Invalid positive scenario parameter {key}.{field}")
        if not 0 <= profile["flood_peak_day"] < 366:
            raise ValueError(f"Invalid flood peak day: {key}")
        if profile["source_status"] != "synthetic" or profile["parameter_status"] != "illustrative_uncalibrated":
            raise ValueError("Registry entries describe uncalibrated synthetic scenarios only")
    return profiles


def reach_geometry(profile: dict, n_nodes: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand scalar scenario geometry into wider, flatter downstream reaches.

    Multipliers are illustrative assumptions shared by generator and regularizer;
    they are not bathymetric observations. Manning hydraulic radius uses a
    rectangular cross-section, R = (width * depth) / (width + 2 * depth).
    """
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive")
    widths = float(profile["width_m"]) * np.linspace(1.0, 1.28, n_nodes)
    slopes = float(profile["slope"]) * np.linspace(1.0, 0.72, n_nodes)
    roughness = np.full(n_nodes, float(profile["manning_n"]))
    return widths, slopes, roughness


def manning_discharge(depth, widths, slopes, roughness):
    """Rectangular-channel Manning relation, SI units (USGS source: manning)."""
    depth = np.maximum(depth, 0.0)
    area = widths * depth
    radius = area / (widths + 2 * depth)
    return area * np.power(radius, 2.0 / 3.0) * np.sqrt(slopes) / roughness


def depth_from_discharge(discharge, widths, slopes, roughness):
    """Monotonic bisection of the same full rectangular Manning relation."""
    q = np.asarray(discharge, dtype=np.float64)
    low = np.zeros_like(q)
    high = np.full_like(q, 512.0)
    if np.any(manning_discharge(high, widths, slopes, roughness) < q):
        raise ValueError("Discharge exceeds the illustrative depth search bracket")
    for _ in range(38):
        middle = (low + high) / 2
        less = manning_discharge(middle, widths, slopes, roughness) < q
        low = np.where(less, middle, low)
        high = np.where(less, high, middle)
    return (low + high) / 2


def _season(timestamp: datetime) -> float:
    year_start = timestamp.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    days = (timestamp - year_start).total_seconds() / 86400
    return float(np.sin(2 * np.pi * days / (366 if calendar.isleap(timestamp.year) else 365)))


def generate_series(profile: dict, hours: int = 8760, seed: int = 42) -> tuple[np.ndarray, list[str]]:
    """Generate [hour, 5 reaches, 6 features] from 2024-01-01 00:00 UTC.

    Each time step's forcing is drawn together, so extending the requested series
    preserves its prefix. Runoff filters, reach routing and sediment hysteresis
    use only current/past values. The annual flood pulse is an explicit imposed
    scenario; future rainfall is never supplied to a forecasting model.
    """
    if not isinstance(hours, (int, np.integer)) or hours < 1:
        raise ValueError("hours must be a positive integer")
    n_nodes = 5
    rng = np.random.default_rng(river_seed(profile["id"], seed))
    # Time-major draws keep prefixes identical when hours changes.
    random = rng.random((hours, n_nodes, 4))
    days = np.arange(hours) / 24.0
    peak = float(profile["flood_peak_day"])
    amplitude = float(profile.get("seasonal_amplitude", 0.3))
    seasonal = 1 + amplitude * np.cos(2 * np.pi * (days - peak) / 366)
    flood = np.exp(-0.5 * ((days - peak) / 5.5) ** 2)
    wetness = np.clip(0.09 + 0.12 * (seasonal - (1 - amplitude)) / max(2 * amplitude, 1e-6), 0.04, 0.28)
    rain_scale = float(profile.get("rainfall_scale", 2.0))
    rain = (random[:, :, 0] < wetness[:, None]) * (-np.log(np.clip(random[:, :, 1], 1e-8, 1.0))) * rain_scale
    rain += rain_scale * 1.5 * flood[:, None] * np.linspace(1.0, 0.65, n_nodes)
    base = float(profile["base_discharge"])
    runoff = np.zeros(n_nodes)
    discharge = np.empty((hours, n_nodes), dtype=np.float64)
    sediment = np.empty_like(discharge)
    lag = int(round(float(profile["lag_hours"])))
    response = float(profile.get("runoff_response", 0.2))
    previous_q = np.full(n_nodes, base)
    previous_sediment = np.full(n_nodes, float(profile["sediment_scale"]))
    for t in range(hours):
        runoff = 0.94 * runoff + 0.06 * rain[t]
        upstream = base * (seasonal[t] + response * runoff[0] / max(rain_scale, 0.01))
        upstream *= 1 + 0.014 * (random[t, 0, 2] - 0.5)
        discharge[t, 0] = max(base * 0.15, upstream)
        for node in range(1, n_nodes):
            routed = discharge[max(0, t - lag), node - 1]
            local = base * (0.055 * seasonal[t] + 0.04 * response * runoff[node] / max(rain_scale, 0.01))
            incoming = 0.995 * routed + local
            discharge[t, node] = 0.72 * previous_q[node] + 0.28 * incoming
        relative_q = discharge[t] / base
        rising = (discharge[t] - previous_q) / base
        # Same discharge can have distinct concentrations on rising/falling limbs.
        equilibrium = float(profile["sediment_scale"]) * (0.22 + 0.65 * relative_q**1.35)
        equilibrium *= np.exp(np.clip(10 * rising, -0.25, 0.45))
        equilibrium *= 1 + 0.04 * (random[t, :, 3] - 0.5)
        sediment[t] = 0.88 * previous_sediment + 0.12 * equilibrium
        previous_q = discharge[t].copy()
        previous_sediment = sediment[t].copy()
    widths, slopes, roughness = reach_geometry(profile)
    depth = depth_from_discharge(discharge, widths, slopes, roughness)
    temp_mean = float(profile.get("temperature_mean", 16))
    temp_amplitude = float(profile.get("temperature_amplitude", 10))
    temperature = temp_mean + temp_amplitude * np.cos(2 * np.pi * (days - 205) / 366)
    temperature = temperature[:, None] + 0.8 * np.sin(2 * np.pi * np.arange(hours)[:, None] / 24) + 0.2 * (random[:, :, 2] - 0.5)
    dates = [START + timedelta(hours=int(i)) for i in range(hours)]
    season = np.array([_season(date) for date in dates])[:, None] * np.ones((1, n_nodes))
    series = np.stack((depth, discharge, sediment, temperature, rain, season), axis=-1).astype(np.float32)
    timestamps = [date.isoformat().replace("+00:00", "Z") for date in dates]
    return series, timestamps


def chronological_windows(series: np.ndarray, seq_len: int = 24, horizon: int = 6, stride: int = 24) -> dict:
    """70/15/15 chronological splits with disjoint target timestamps.

    Boundaries are exclusive raw-series indices. Earlier observations can appear
    as validation/test input context. Windows whose targets straddle a boundary
    are excluded. Normalization must be fitted on series[:train_end] only.
    """
    series = np.asarray(series)
    if series.ndim != 3 or series.shape[1:] != (5, 6) or not np.isfinite(series).all():
        raise ValueError("Expected finite [time, 5, 6] river series")
    if min(seq_len, horizon, stride) < 1 or any(not isinstance(value, (int, np.integer)) for value in (seq_len, horizon, stride)):
        raise ValueError("Sequence length, horizon and stride must be positive integers")
    total = len(series)
    train_end, val_end = int(total * 0.70), int(total * 0.85)
    if train_end < seq_len + horizon or total - val_end < horizon:
        raise ValueError("Series is too short for all chronological splits")
    groups = {name: [] for name in ("train", "val", "test")}
    for origin in range(seq_len - 1, total - horizon, stride):
        first_target, last_target = origin + 1, origin + horizon
        if last_target < train_end:
            groups["train"].append(origin)
        elif first_target >= train_end and last_target < val_end:
            groups["val"].append(origin)
        elif first_target >= val_end:
            groups["test"].append(origin)
    result = {"train_end": train_end, "val_end": val_end, "boundaries": {"train": [0, train_end], "val": [train_end, val_end], "test": [val_end, total]}}
    for name, origins in groups.items():
        if not origins:
            raise ValueError(f"No {name} windows; use a longer series or smaller stride")
        result[name] = {"x": np.stack([series[i - seq_len + 1:i + 1] for i in origins]), "y": np.stack([series[i + 1:i + horizon + 1, :, :3] for i in origins]), "origins": np.asarray(origins, dtype=np.int64)}
    return result


def load_observed_csv(path: Path, profile: dict) -> tuple[np.ndarray, list[str]]:
    """Import complete hourly observations without imputation or interpolation.

    Required columns: timestamp, station_id, water_depth, discharge, sediment,
    temperature, rainfall. Depth is metres above the local bed, not datum stage;
    concentration is kg/m³. Reach IDs must match profile.station_ids (r1..r5).
    ISO timestamps normalize to UTC; timezone-free timestamps are treated as UTC.
    Depth, discharge, sediment and rainfall cannot be negative; °C can be signed.
    Every hour must contain every reach once. Any gap/duplicate/nonfinite value
    raises ValueError instead of filling with a future observation.
    """
    required = {"timestamp", "station_id", *FEATURE_NAMES[:5]}
    stations = profile.get("station_ids", [f"r{i + 1}" for i in range(5)])
    if len(stations) != 5 or len(set(stations)) != 5:
        raise ValueError("Expected five distinct configured reach IDs")
    positions = {station: i for i, station in enumerate(stations)}
    samples: dict[datetime, dict[str, list[float]]] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("CSV contains duplicate column headers")
        if required - set(fields):
            raise ValueError(f"CSV missing required columns: {sorted(required - set(fields))}")
        for line, row in enumerate(reader, start=2):
            try:
                if None in row:
                    raise ValueError("Row has more values than CSV headers")
                timestamp = datetime.fromisoformat(row["timestamp"].strip().replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                timestamp = timestamp.astimezone(timezone.utc)
                if timestamp.minute or timestamp.second or timestamp.microsecond:
                    raise ValueError("Timestamps must lie on full UTC hours")
                station = row["station_id"].strip()
                if station not in positions:
                    raise ValueError(f"Unknown station_id {station}")
                values = [float(row[feature]) for feature in FEATURE_NAMES[:5]]
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite observation")
                if any(values[index] < 0 for index in (0, 1, 2, 4)):
                    raise ValueError("Negative depth, discharge, sediment or rainfall")
                same_hour = samples.setdefault(timestamp, {})
                if station in same_hour:
                    raise ValueError("Duplicate timestamp/station pair")
                same_hour[station] = values
            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError(f"Invalid CSV row {line}: {error}") from error
    if not samples:
        raise ValueError("CSV contains no observations")
    dates = sorted(samples)
    for before, after in zip(dates, dates[1:]):
        if after - before != timedelta(hours=1):
            raise ValueError("Missing or irregular hourly timestamps; no samples were filled")
    series = np.empty((len(dates), 5, 6), dtype=np.float32)
    for index, timestamp in enumerate(dates):
        if set(samples[timestamp]) != set(stations):
            raise ValueError(f"Missing reach observations at {timestamp.isoformat()}")
        for station, position in positions.items():
            series[index, position, :5] = samples[timestamp][station]
            series[index, position, 5] = _season(timestamp)
    if not np.isfinite(series).all():
        raise ValueError("Observation values overflow float32 storage")
    return series, [date.isoformat().replace("+00:00", "Z") for date in dates]
