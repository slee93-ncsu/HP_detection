"""
Input adapters: turn a utility's meter data into the hourly frame the
feature code expects.

Two input layouts are supported:

    resstock  a directory of {bldg_id}-{upgrade}.parquet files, one per
              building, in the ResStock schema used for training
    long      one CSV or parquet table with one row per meter per interval
              (meter id, timestamp, load, optionally temperature)

Outdoor temperature can be a column of the load table or a separate
weather file, optionally keyed by station so each meter gets its own
series.

Every layout ends in to_hourly(), which reproduces the resampling used
in training: timestamps are interval-end, load is summed per hour, and
temperature is averaged per hour.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

# ResStock schema used for training.
RESSTOCK_TIMESTAMP = "timestamp"
RESSTOCK_LOAD = "out.electricity.total.energy_consumption..kwh"
RESSTOCK_TEMP = "out.outdoor_air_drybulb_temp..c"
RESSTOCK_FILENAME = re.compile(r"^(\d+)-\d+\.parquet$")

LOAD_UNITS = ("kWh", "Wh", "kW")
TEMP_UNITS = ("C", "F")
CONVENTIONS = ("end", "start")


@dataclass
class InputConfig:
    """Describes the utility's data layout. Every field can be set in the
    YAML/JSON config file or on the command line."""

    format: str = "auto"                    # auto | resstock | long

    id_column: str = "meter_id"
    timestamp_column: str = "timestamp"
    load_column: str = "kwh"
    load_unit: str = "kWh"                  # kWh | Wh | kW (average demand)
    temp_column: Optional[str] = None       # if temperature is in the load table
    temp_unit: str = "C"                    # C | F (training data: C)
    timestamp_convention: str = "end"       # end | start of the interval
    timezone: Optional[str] = None          # e.g. America/Chicago

    weather_file: Optional[str] = None
    weather_timestamp_column: str = "timestamp"
    weather_temp_column: str = "temp"
    weather_key_column: Optional[str] = None  # e.g. station_id, in both files

    @classmethod
    def from_dict(cls, values: dict) -> "InputConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**values)

    def validate(self) -> None:
        if self.load_unit not in LOAD_UNITS:
            raise ValueError(f"load_unit must be one of {LOAD_UNITS}")
        if self.temp_unit not in TEMP_UNITS:
            raise ValueError(f"temp_unit must be one of {TEMP_UNITS}")
        if self.timestamp_convention not in CONVENTIONS:
            raise ValueError(f"timestamp_convention must be one of {CONVENTIONS}")


def to_fahrenheit(celsius):
    return celsius * 9.0 / 5.0 + 32.0


def to_celsius(fahrenheit):
    return (fahrenheit - 32.0) * 5.0 / 9.0


def to_hourly(load_kwh: pd.Series, temp_c: pd.Series) -> pd.DataFrame:
    """Resample interval data to the hourly frame used by the feature code.

    load_kwh  energy per interval (kWh), indexed by interval-end timestamp
    temp_c    outdoor temperature (C), indexed by timestamp; may have a
              different index than load_kwh (e.g. a separate weather file)

    Hours with no load readings stay NaN instead of becoming 0 (min_count=1).
    Training data had no gaps, so this does not change training features.
    """
    load_kwh = load_kwh.sort_index()
    temp_c = temp_c.sort_index()

    load_h = load_kwh.resample("1h").sum(min_count=1)
    temp_h = temp_c.resample("1h").mean().reindex(load_h.index)

    hourly = pd.DataFrame({"load": load_h, "temp_c": temp_h})
    hourly["temp_f"] = to_fahrenheit(hourly["temp_c"])
    hourly["hour"] = hourly.index.hour
    hourly["month"] = hourly.index.month
    hourly["date"] = hourly.index.date

    return hourly


# --------------------------------------------------------------------------
# Timestamp and unit normalization for the long layout
# --------------------------------------------------------------------------

def _parse_timestamps(values: pd.Series, timezone: Optional[str]) -> pd.DatetimeIndex:
    """Parse timestamps to naive local standard time (no DST), as in ResStock.

    Timestamps carrying a UTC offset need `timezone`: they are shifted to
    that zone's standard (winter) offset for the whole year, so there are
    no DST gaps or repeated hours. Naive timestamps are used as given.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        parsed = pd.to_datetime(values)

    if not pd.api.types.is_datetime64_any_dtype(parsed):
        # Mixed UTC offsets (e.g. -05:00 and -06:00 across DST).
        parsed = pd.to_datetime(values, utc=True)

    ts = pd.DatetimeIndex(parsed)

    if ts.tz is None:
        return ts

    if not timezone:
        raise ValueError("Timestamps carry a UTC offset; set timezone "
                         "(e.g. America/Chicago) so hours are local.")

    standard_offset = pd.Timestamp("2021-01-15", tz=timezone).utcoffset()
    return ts.tz_convert("UTC").tz_localize(None) + standard_offset


def _interval(index: pd.DatetimeIndex) -> pd.Timedelta:
    """Typical reading interval (median spacing)."""
    diffs = index.sort_values().to_series().diff().dropna()
    if diffs.empty:
        raise ValueError("Need at least two readings to infer the interval.")
    return diffs.median()


def _normalize_meter(frame: pd.DataFrame, cfg: InputConfig) -> tuple[pd.Series, Optional[pd.Series], dict]:
    """Return (load_kwh, temp_c or None, raw_info) for one meter's rows."""
    index = _parse_timestamps(frame[cfg.timestamp_column], cfg.timezone)
    interval = _interval(index)

    if cfg.timestamp_convention == "start":
        index = index + interval  # label each reading by interval end

    load = pd.Series(frame[cfg.load_column].to_numpy(dtype=float), index=index)
    if cfg.load_unit == "Wh":
        load = load / 1000.0
    elif cfg.load_unit == "kW":
        load = load * (interval / pd.Timedelta(hours=1))

    temp = None
    if cfg.temp_column:
        temp = pd.Series(frame[cfg.temp_column].to_numpy(dtype=float), index=index)
        if cfg.temp_unit == "F":
            temp = to_celsius(temp)

    raw_info = {
        "interval_minutes": interval / pd.Timedelta(minutes=1),
        "n_duplicate_timestamps": int(index.duplicated().sum()),
    }

    return load, temp, raw_info


def _read_table(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _load_weather(cfg: InputConfig) -> dict:
    """Return {key: temp_c Series}; key is None when there is one series."""
    weather = _read_table(cfg.weather_file)
    index = _parse_timestamps(weather[cfg.weather_timestamp_column], cfg.timezone)

    temp = weather[cfg.weather_temp_column].to_numpy(dtype=float)
    if cfg.temp_unit == "F":
        temp = to_celsius(temp)

    if cfg.weather_key_column is None:
        return {None: pd.Series(temp, index=index)}

    series = pd.Series(temp, index=index)
    keys = weather[cfg.weather_key_column].to_numpy()
    return {key: series[keys == key] for key in pd.unique(keys)}


# --------------------------------------------------------------------------
# Iterators yielding (building_id, hourly frame, raw_info)
# --------------------------------------------------------------------------

def iter_resstock_dir(directory) -> Iterator[tuple]:
    """Yield buildings from a directory of ResStock-schema parquet files."""
    for path in sorted(Path(directory).glob("*.parquet")):
        match = RESSTOCK_FILENAME.match(path.name)
        if match is None:
            continue
        yield int(match.group(1)), path


def read_resstock_file(path) -> tuple[pd.DataFrame, dict]:
    """Read one ResStock parquet file exactly as training did."""
    frame = pd.read_parquet(
        path, columns=[RESSTOCK_TIMESTAMP, RESSTOCK_LOAD, RESSTOCK_TEMP],
    ).set_index(RESSTOCK_TIMESTAMP).sort_index()

    raw_info = {
        "interval_minutes": _interval(frame.index) / pd.Timedelta(minutes=1),
        "n_duplicate_timestamps": int(frame.index.duplicated().sum()),
    }
    return to_hourly(frame[RESSTOCK_LOAD], frame[RESSTOCK_TEMP]), raw_info


def iter_long_table(path, cfg: InputConfig) -> Iterator[tuple]:
    """Yield (meter_id, hourly, raw_info) from a long-format table."""
    table = _read_table(path)

    required = [cfg.id_column, cfg.timestamp_column, cfg.load_column]
    if cfg.temp_column:
        required.append(cfg.temp_column)
    if cfg.weather_key_column:
        required.append(cfg.weather_key_column)
    missing = [c for c in required if c not in table.columns]
    if missing:
        raise ValueError(f"Columns not found in {path}: {missing}. "
                         f"Available: {list(table.columns)[:20]}")

    if not cfg.temp_column and not cfg.weather_file:
        raise ValueError("No temperature source: set temp_column or weather_file.")

    weather = _load_weather(cfg) if cfg.weather_file else None

    for meter_id, rows in table.groupby(cfg.id_column, sort=True):
        load, temp, raw_info = _normalize_meter(rows, cfg)

        if weather is not None:
            key = rows[cfg.weather_key_column].iloc[0] if cfg.weather_key_column else None
            if key not in weather:
                raise ValueError(f"No weather series for {cfg.weather_key_column}={key!r} "
                                 f"(meter {meter_id}).")
            temp = weather[key]

        yield meter_id, to_hourly(load, temp), raw_info


def detect_format(input_path) -> str:
    path = Path(input_path)
    if path.is_dir():
        if any(RESSTOCK_FILENAME.match(p.name) for p in path.glob("*.parquet")):
            return "resstock"
        raise ValueError(f"{path} is a directory but has no {{bldg_id}}-{{upgrade}}.parquet files.")
    return "long"
