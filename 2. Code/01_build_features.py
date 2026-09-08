"""
Extract 102 statistical features per building from ResStock parquet files.

Input : {bldg_id}-{upgrade}.parquet, 15-min timeseries, one file per building.
        Columns read: timestamp, total electricity (kWh), outdoor temp (C).
Output: dallas_features_groupA_E.csv, one row per building,
        bldg_id + 102 feature columns (A=25, B=22, C=8, D=37, E=10).

Data is resampled to hourly and temperature converted to Fahrenheit before
any feature is computed. All temperature thresholds are in Fahrenheit.

Usage:
    python 01_build_features.py --parquet-dir "path/to/parquet" --workers 8
    python 01_build_features.py --parquet-dir "path/to/parquet" --limit 10
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

EXPECTED_FEATURE_COUNT = 102
FILENAME_PATTERN = re.compile(r"^(\d+)-\d+\.parquet$")

SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}

MORNING_HOURS = [6, 7, 8, 9]
EVENING_HOURS = [17, 18, 19, 20, 21]
OVERNIGHT_HOURS = [0, 1, 2, 3, 4]
MIDDAY_HOURS = [11, 12, 13, 14]

# Zone edges in Fahrenheit: cold <40, mild 40-60, warm 60-80, hot >=80.
ZONE_EDGES = [(-np.inf, 40), (40, 60), (60, 80), (80, np.inf)]
ZONE_NAMES = ["cold", "mild", "warm", "hot"]


def to_fahrenheit(celsius):
    return celsius * 9.0 / 5.0 + 32.0


def safe_ratio(numerator, denominator):
    """Return numerator/denominator, or NaN if the denominator is 0 or NaN."""
    if pd.notna(numerator) and pd.notna(denominator) and denominator != 0:
        return numerator / denominator
    return np.nan


def load_building(parquet_path) -> pd.DataFrame:
    """Read one parquet file and return an hourly DataFrame.

    Returns 8760 rows indexed by timestamp with columns:
        load    kWh summed over the hour
        temp_c  outdoor temperature averaged over the hour
        temp_f  temp_c in Fahrenheit
        hour, month, date
    """
    frame = pd.read_parquet(
        parquet_path,
        columns=[COL_TIMESTAMP, COL_TOTAL_LOAD, COL_OUTDOOR_TEMP],
    ).set_index(COL_TIMESTAMP).sort_index()

    hourly = pd.DataFrame({
        "load": frame[COL_TOTAL_LOAD].resample("1h").sum(),
        "temp_c": frame[COL_OUTDOOR_TEMP].resample("1h").mean(),
    })

    hourly["temp_f"] = to_fahrenheit(hourly["temp_c"])
    hourly["hour"] = hourly.index.hour
    hourly["month"] = hourly.index.month
    hourly["date"] = hourly.index.date

    return hourly


def group_a_annual(hourly: pd.DataFrame) -> dict:
    """Group A (25): annual load statistics, percentiles, and dispersion."""
    load = hourly["load"]
    feats = {}

    feats["A_annual_mean_kwh"] = load.mean()
    feats["A_annual_std_kwh"] = load.std()
    feats["A_annual_sum_kwh"] = load.sum()

    for q in (10, 25, 50, 75, 90, 95, 99):
        feats[f"A_p{q}"] = load.quantile(q / 100)

    feats["A_max_kwh"] = load.max()
    feats["A_min_kwh"] = load.min()
    feats["A_cv"] = safe_ratio(load.std(), load.mean())
    feats["A_load_factor"] = safe_ratio(load.mean(), load.max())
    feats["A_skew"] = load.skew()
    feats["A_kurtosis"] = load.kurt()
    feats["A_p05"] = load.quantile(0.05)
    feats["A_p99_p01_ratio"] = safe_ratio(load.quantile(0.99), load.quantile(0.01))
    feats["A_zero_hours_ratio"] = (load == 0).sum() / len(load)
    feats["A_above_mean_ratio"] = (load > load.mean()).sum() / len(load)
    feats["A_range_kwh"] = load.max() - load.min()
    feats["A_p01"] = load.quantile(0.01)
    feats["A_p15"] = load.quantile(0.15)
    feats["A_p85"] = load.quantile(0.85)
    feats["A_mad"] = (load - load.mean()).abs().mean()

    return feats


def group_b_seasonal(hourly: pd.DataFrame) -> dict:
    """Group B (22): per-season mean and std, two season ratios, 12 monthly means.

    Seasons follow SEASON_MAP (four seasons). Script 02 uses a different
    three-season map; the two are independent.
    """
    feats = {}

    hourly = hourly.copy()
    hourly["season"] = hourly["month"].map(SEASON_MAP)

    means = hourly.groupby("season")["load"].mean()
    stds = hourly.groupby("season")["load"].std()

    for season in ("winter", "spring", "summer", "fall"):
        feats[f"B_{season}_mean_kwh"] = means.get(season, np.nan)
        feats[f"B_{season}_std_kwh"] = stds.get(season, np.nan)

    winter = means.get("winter", np.nan)
    summer = means.get("summer", np.nan)
    shoulder = np.nanmean([means.get("spring", np.nan), means.get("fall", np.nan)])

    feats["B_winter_summer_ratio"] = safe_ratio(winter, summer)
    feats["B_seasonal_amplitude"] = (
        max(winter, summer) - shoulder
        if pd.notna(winter) and pd.notna(summer) and pd.notna(shoulder)
        else np.nan
    )

    month_means = hourly.groupby("month")["load"].mean()

    for month in range(1, 13):
        feats[f"B_month{month:02d}_mean_kwh"] = month_means.get(month, np.nan)

    return feats


def group_c_time_of_use(hourly: pd.DataFrame) -> dict:
    """Group C (8): mean load in four time-of-day windows, normalized by the
    all-hours mean, plus peak/trough hour of the average daily profile."""
    feats = {}

    hour_means = hourly.groupby("hour")["load"].mean()
    daily_mean = hour_means.mean()

    morning = hour_means.loc[MORNING_HOURS].mean()
    evening = hour_means.loc[EVENING_HOURS].mean()
    overnight = hour_means.loc[OVERNIGHT_HOURS].mean()
    midday = hour_means.loc[MIDDAY_HOURS].mean()

    feats["C_morning_peak_ratio"] = safe_ratio(morning, daily_mean)
    feats["C_evening_peak_ratio"] = safe_ratio(evening, daily_mean)
    feats["C_overnight_ratio"] = safe_ratio(overnight, daily_mean)
    feats["C_midday_ratio"] = safe_ratio(midday, daily_mean)
    feats["C_evening_morning_ratio"] = safe_ratio(evening, morning)

    feats["C_peak_hour"] = hour_means.idxmax()
    feats["C_peak_hour_value"] = hour_means.max()
    feats["C_trough_hour"] = hour_means.idxmin()

    return feats


def group_d_temperature(hourly: pd.DataFrame) -> dict:
    """Group D (37): 8 statistics for each of 4 temperature zones,
    4 cross-zone mean ratios, and the all-hours load-temperature correlation.

    Statistics with too few hours to define (std, autocorr, correlation)
    are set to NaN rather than dropped.
    """
    feats = {}

    valid = hourly.dropna(subset=["temp_f", "load"]).copy()
    valid["load_ramp"] = valid["load"].diff()

    zone_means = {}

    for name, (low, high) in zip(ZONE_NAMES, ZONE_EDGES):
        zone = valid[(valid["temp_f"] >= low) & (valid["temp_f"] < high)]
        n = len(zone)

        zone_means[name] = zone["load"].mean() if n > 0 else np.nan

        feats[f"D_{name}_n_hours"] = n
        feats[f"D_{name}_mean_kwh"] = zone_means[name]
        feats[f"D_{name}_std_kwh"] = zone["load"].std() if n > 1 else np.nan
        feats[f"D_{name}_max_kwh"] = zone["load"].max() if n > 0 else np.nan
        feats[f"D_{name}_ramp_std"] = zone["load_ramp"].std() if n > 1 else np.nan
        feats[f"D_{name}_ramp_abs_mean"] = (
            zone["load_ramp"].abs().mean() if n > 0 else np.nan)
        feats[f"D_{name}_autocorr_1h"] = (
            zone["load"].autocorr(lag=1) if n > 2 else np.nan)
        feats[f"D_{name}_temp_load_corr"] = (
            zone["temp_f"].corr(zone["load"]) if n > 2 else np.nan)

    feats["D_ratio_cold_over_warm"] = safe_ratio(zone_means["cold"], zone_means["warm"])
    feats["D_ratio_cold_over_hot"] = safe_ratio(zone_means["cold"], zone_means["hot"])
    feats["D_ratio_mild_over_warm"] = safe_ratio(zone_means["mild"], zone_means["warm"])
    feats["D_ratio_hot_over_cold"] = safe_ratio(zone_means["hot"], zone_means["cold"])

    feats["D_temp_load_corr_all"] = valid["temp_f"].corr(valid["load"])

    return feats


def binned_means(valid: pd.DataFrame) -> pd.Series:
    """Mean load per 5 F temperature bin, indexed by bin lower edge.

    Group E slopes are fitted on these bin means rather than raw hourly
    points so that each 5 F band contributes equally regardless of how
    many hours fall in it.
    """
    binned = valid.copy()
    binned["temp_bin"] = (binned["temp_f"] // 5) * 5

    return binned.groupby("temp_bin")["load"].mean().sort_index()


def segment_slope(bins: pd.Series, low: float, high: float) -> float:
    """Least-squares slope (kWh per degree F) of bin means over [low, high].

    Returns NaN if fewer than 2 bins fall in the range.
    """
    segment = bins[(bins.index >= low) & (bins.index <= high)]

    if len(segment) < 2:
        return np.nan

    return np.polyfit(segment.index.values.astype(float),
                      segment.values.astype(float), 1)[0]


def group_e_signature(hourly: pd.DataFrame) -> dict:
    """Group E (10): load-temperature slopes over six ranges, their ratio,
    the 35-40 F minus 40-45 F bin-mean difference, and two spread measures."""
    feats = {}

    valid = hourly.dropna(subset=["temp_f", "load"]).copy()
    bins = binned_means(valid)
    load = hourly["load"]

    feats["E_slope_cold"] = segment_slope(bins, -20, 40)
    feats["E_slope_mild"] = segment_slope(bins, 40, 60)
    feats["E_slope_warm"] = segment_slope(bins, 60, 80)
    feats["E_slope_hot"] = segment_slope(bins, 80, 110)

    slope_40_65 = segment_slope(bins, 40, 65)
    slope_0_40 = segment_slope(bins, 0, 40)

    feats["E_slope_zone40_65F"] = slope_40_65
    feats["E_slope_zone0_40F"] = slope_0_40
    feats["E_slope_kink_ratio"] = safe_ratio(slope_0_40, slope_40_65)

    above = bins[(bins.index >= 40) & (bins.index < 45)]
    below = bins[(bins.index >= 35) & (bins.index < 40)]

    feats["E_zone40F_jump"] = (
        below.mean() - above.mean() if len(above) and len(below) else np.nan)

    feats["E_power_iqr"] = load.quantile(0.75) - load.quantile(0.25)
    feats["E_power_p90_p10_ratio"] = safe_ratio(load.quantile(0.90),
                                                load.quantile(0.10))

    return feats


def compute_features(parquet_path, bldg_id: int) -> dict:
    """Return {'bldg_id': id, ...102 features} for one building."""
    hourly = load_building(parquet_path)

    feats = {"bldg_id": bldg_id}
    feats.update(group_a_annual(hourly))
    feats.update(group_b_seasonal(hourly))
    feats.update(group_c_time_of_use(hourly))
    feats.update(group_d_temperature(hourly))
    feats.update(group_e_signature(hourly))

    return feats


def process_one(path_str: str):
    """Wrapper for parallel execution. Returns None if the filename does not
    match the expected pattern or the file cannot be processed."""
    path = Path(path_str)
    match = FILENAME_PATTERN.match(path.name)

    if match is None:
        return None

    try:
        return compute_features(path, int(match.group(1)))
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
    output_path = work_dir / "dallas_features_groupA_E.csv"
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

    # NaN is written as-is; script 03 converts NaN and inf to 0 after merging.
    features = pd.DataFrame(rows).sort_values("bldg_id")
    features.to_csv(output_path, index=False, encoding="utf-8-sig")

    n_features = len(features.columns) - 1

    print(f"\nBuildings processed  : {len(features):,}")
    print(f"Buildings failed     : {n_failed:,}")
    print(f"Features per building: {n_features}")

    for group in "ABCDE":
        n = sum(c.startswith(f"{group}_") for c in features.columns)
        print(f"  Group {group}: {n}")

    if n_features != EXPECTED_FEATURE_COUNT:
        raise SystemExit(
            f"\nExpected {EXPECTED_FEATURE_COUNT} features, generated {n_features}.")

    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
