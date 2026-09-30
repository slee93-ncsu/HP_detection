"""
Extract four 24-hour load profiles per building from ResStock parquet files.

Input : {bldg_id}-{upgrade}.parquet, same files as script 01.
Output: dallas_cnn_profiles.csv, one row per building,
        bldg_id + 96 columns named "{profile}_h{hour:02d}".

Profiles:
    winter_weekday    mean hourly load, weekdays in Dec/Jan/Feb
    summer_weekday    mean hourly load, weekdays in Jun/Jul/Aug
    shoulder_weekday  mean hourly load, weekdays in Mar-May and Sep-Nov
    peak_day          hourly load on the highest daily-total day (not
                      weekday-filtered)

Column order is profile-major: 24 columns of winter_weekday, then
summer_weekday, then shoulder_weekday, then peak_day. Script 03 reshapes
this to (n_buildings, 4, 24) and depends on that order.

The three-season map here is separate from the four-season map in
script 01; the CNN takes exactly four input channels.

Usage:
    python 02_build_profiles.py --parquet-dir "path/to/parquet" --workers 8
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

WORK_DIR = Path(os.environ.get("HP_WORK_DIR", "outputs"))
PARQUET_DIR = Path(os.environ.get("HP_DATA_ROOT", "data")) / "timeseries"

COL_TIMESTAMP = "timestamp"
COL_TOTAL_LOAD = "out.electricity.total.energy_consumption..kwh"
COL_OUTDOOR_TEMP = "out.outdoor_air_drybulb_temp..c"

FILENAME_PATTERN = re.compile(r"^(\d+)-\d+\.parquet$")

PROFILE_NAMES = ("winter_weekday", "summer_weekday", "shoulder_weekday", "peak_day")
PROFILE_LENGTH = 24
EXPECTED_COLUMNS = len(PROFILE_NAMES) * PROFILE_LENGTH  # 96

SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3: "shoulder", 4: "shoulder", 5: "shoulder",
    6: "summer", 7: "summer", 8: "summer",
    9: "shoulder", 10: "shoulder", 11: "shoulder",
}


def load_building(parquet_path) -> pd.DataFrame:
    """Read one parquet file and return an hourly DataFrame with columns
    load, temp_c, hour, month, date. Same resampling as script 01."""
    frame = pd.read_parquet(
        parquet_path,
        columns=[COL_TIMESTAMP, COL_TOTAL_LOAD, COL_OUTDOOR_TEMP],
    ).set_index(COL_TIMESTAMP).sort_index()

    hourly = pd.DataFrame({
        "load": frame[COL_TOTAL_LOAD].resample("1h").sum(),
        "temp_c": frame[COL_OUTDOOR_TEMP].resample("1h").mean(),
    })

    hourly["hour"] = hourly.index.hour
    hourly["month"] = hourly.index.month
    hourly["date"] = hourly.index.date

    return hourly


def average_24h(subset: pd.DataFrame) -> list:
    """Mean load per hour of day over the given rows.

    Returns a list of 24 floats. Hours with no rows become NaN; an empty
    subset returns 24 NaNs.
    """
    if len(subset) == 0:
        return [np.nan] * PROFILE_LENGTH

    profile = subset.groupby("hour")["load"].mean()

    return [profile.get(h, np.nan) for h in range(PROFILE_LENGTH)]


def compute_profiles(hourly: pd.DataFrame) -> dict:
    """Return {profile_name: list of 24 floats} for the four profiles."""
    frame = hourly.copy()
    frame["is_weekday"] = frame.index.weekday < 5  # 0=Mon .. 6=Sun
    frame["season"] = frame["month"].map(SEASON_MAP)

    daily_total = frame.groupby("date")["load"].sum()

    if len(daily_total) > 0:
        peak_day = frame[frame["date"] == daily_total.idxmax()]
        peak_profile = (peak_day.set_index("hour")["load"]
                        .reindex(range(PROFILE_LENGTH)).tolist())
    else:
        peak_profile = [np.nan] * PROFILE_LENGTH

    def seasonal(season):
        return average_24h(frame[(frame["season"] == season) & frame["is_weekday"]])

    return {
        "winter_weekday": seasonal("winter"),
        "summer_weekday": seasonal("summer"),
        "shoulder_weekday": seasonal("shoulder"),
        "peak_day": peak_profile,
    }


def profile_columns() -> list:
    """The 96 column names in profile-major order."""
    return [f"{name}_h{h:02d}" for name in PROFILE_NAMES for h in range(PROFILE_LENGTH)]


def process_one(path_str: str):
    """Wrapper for parallel execution. Returns a flat dict of one row, or
    None if the filename does not match or the file cannot be processed."""
    path = Path(path_str)
    match = FILENAME_PATTERN.match(path.name)

    if match is None:
        return None

    bldg_id = int(match.group(1))

    try:
        profiles = compute_profiles(load_building(path))

        row = {"bldg_id": bldg_id}
        for name, values in profiles.items():
            for hour, value in enumerate(values):
                row[f"{name}_h{hour:02d}"] = value

        return row
    except Exception as error:  # noqa: BLE001
        print(f"  FAILED {path.name}: {error}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", default=str(PARQUET_DIR))
    parser.add_argument("--work-dir", default=str(WORK_DIR))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = work_dir / "dallas_cnn_profiles.csv"
    print(f"Output folder: {work_dir}")

    paths = sorted(Path(args.parquet_dir).glob("*.parquet"))

    if not paths:
        raise SystemExit(f"No parquet files found in {args.parquet_dir}")

    if args.limit:
        paths = paths[: args.limit]

    print(f"Processing {len(paths):,} buildings from {args.parquet_dir}")

    rows, n_failed = [], 0

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_one, str(p)) for p in paths]

            for done, future in enumerate(as_completed(futures), start=1):
                result = future.result()

                if result is None:
                    n_failed += 1
                else:
                    rows.append(result)

                if done % 250 == 0:
                    print(f"  {done:,}/{len(paths):,}")
    else:
        for done, path in enumerate(paths, start=1):
            result = process_one(str(path))

            if result is None:
                n_failed += 1
            else:
                rows.append(result)

            if done % 250 == 0:
                print(f"  {done:,}/{len(paths):,}")

    if not rows:
        raise SystemExit("No buildings processed successfully.")

    # Reindex explicitly rather than relying on dict insertion order.
    profiles = pd.DataFrame(rows).sort_values("bldg_id")
    profiles = profiles[["bldg_id"] + profile_columns()]
    profiles.to_csv(output_path, index=False, encoding="utf-8-sig")

    n_columns = len(profiles.columns) - 1

    print(f"\nBuildings processed : {len(profiles):,}")
    print(f"Buildings failed    : {n_failed:,}")
    print(f"Profile columns     : {n_columns} "
          f"({len(PROFILE_NAMES)} profiles x {PROFILE_LENGTH} hours)")

    if n_columns != EXPECTED_COLUMNS:
        raise SystemExit(f"\nExpected {EXPECTED_COLUMNS} columns, got {n_columns}.")

    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
