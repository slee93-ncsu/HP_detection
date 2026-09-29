"""
Checks that inference reproduces the training pipeline exactly.

1. Features and profiles computed by the package from the sample
   buildings equal the rows produced in training by
   2. Code/01_build_features.py and 02_build_profiles.py
   (stored in tests/data/expected_*.csv).
2. Predictions from raw data equal predictions from those training rows.
3. The same data given as a utility-style long table (interval-start
   timestamps, kW, Fahrenheit, separate weather file, UTC offsets) gives
   the same predictions.

Run from the "3. Model" folder:  pytest
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hp_detection import InputConfig, build_inputs, load_bundle, predict
from hp_detection.profiles import profile_columns

DATA = Path(__file__).resolve().parent / "data"
SAMPLE_DIR = DATA / "resstock_sample"


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


@pytest.fixture(scope="module")
def package_inputs():
    return build_inputs(SAMPLE_DIR, InputConfig(format="resstock"), verbose=False)


@pytest.fixture(scope="module")
def training_rows(package_inputs):
    ids = package_inputs[0].index
    read = lambda name: pd.read_csv(DATA / name).set_index("bldg_id").loc[ids]
    features = read("expected_features.csv")
    features.index.name = "building_id"
    profiles = read("expected_profiles.csv")
    profiles.index.name = "building_id"
    return features, profiles


def test_features_match_training(bundle, package_inputs, training_rows):
    ours = package_inputs[0][bundle["stat_columns"]]
    theirs = training_rows[0][bundle["stat_columns"]]
    # Training CSVs store float32 values rounded to ~8 significant digits.
    pd.testing.assert_frame_equal(ours, theirs, check_dtype=False, rtol=1e-6)


def test_profiles_match_training(package_inputs, training_rows):
    ours = package_inputs[1][profile_columns()]
    theirs = training_rows[1][profile_columns()]
    pd.testing.assert_frame_equal(ours, theirs, check_dtype=False, rtol=1e-6)


def test_predictions_match_training_inputs(bundle, package_inputs, training_rows):
    ours = predict(bundle, package_inputs[0], package_inputs[1])
    theirs = predict(bundle, *training_rows)
    np.testing.assert_allclose(ours["hp_probability"], theirs["hp_probability"], atol=1e-6)
    assert ours["hp_probability"].between(0, 1).all()


def test_sample_quality_is_ok(package_inputs):
    assert (package_inputs[2]["quality_flag"] == "ok").all()


def _utility_style_tables():
    """Rewrite the sample parquet as a long meter table plus a weather file
    with one station per meter, so temperatures stay identical."""
    load_rows, weather_rows = [], []

    for path in sorted(SAMPLE_DIR.glob("*.parquet")):
        bldg_id = int(path.name.split("-")[0])
        frame = pd.read_parquet(path, columns=[
            "timestamp",
            "out.electricity.total.energy_consumption..kwh",
            "out.outdoor_air_drybulb_temp..c",
        ])
        # ResStock is local standard time; express it with a UTC offset.
        end = frame["timestamp"].dt.tz_localize("Etc/GMT+6")
        load_rows.append(pd.DataFrame({
            "meter_id": bldg_id,
            "station_id": f"ST{bldg_id}",
            "read_time": end - pd.Timedelta(minutes=15),         # interval start
            "usage_kw": frame["out.electricity.total.energy_consumption..kwh"] * 4,
        }))
        # Weather timestamps are observation times, not intervals.
        weather_rows.append(pd.DataFrame({
            "station_id": f"ST{bldg_id}",
            "timestamp": end,
            "temp_f": frame["out.outdoor_air_drybulb_temp..c"] * 9 / 5 + 32,
        }))

    return pd.concat(load_rows), pd.concat(weather_rows)


def test_long_format_matches_resstock(bundle, package_inputs, tmp_path):
    load, weather = _utility_style_tables()
    load.to_parquet(tmp_path / "meters.parquet")
    weather.to_csv(tmp_path / "weather.csv", index=False)

    cfg = InputConfig(
        format="long", id_column="meter_id", timestamp_column="read_time",
        load_column="usage_kw", load_unit="kW", timestamp_convention="start",
        timezone="America/Chicago", temp_unit="F",
        weather_file=str(tmp_path / "weather.csv"), weather_temp_column="temp_f",
        weather_key_column="station_id",
    )
    features, profiles, _ = build_inputs(tmp_path / "meters.parquet", cfg, verbose=False)

    ours = predict(bundle, features, profiles)
    reference = predict(bundle, package_inputs[0], package_inputs[1])
    assert list(ours.index) == list(reference.index)
    np.testing.assert_allclose(ours["hp_probability"], reference["hp_probability"], atol=1e-5)
