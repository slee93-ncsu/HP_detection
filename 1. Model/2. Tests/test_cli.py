"""
Runs the command line the way a user would: a meter data CSV, a weather
CSV, and a config file, written from the sample buildings in data/.
Checks the output against predictions made directly from the same data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hp_detection import InputConfig, build_inputs, load_bundle, predict
from hp_detection.cli import main

SAMPLE_DIR = Path(__file__).resolve().parent / "data" / "resstock_sample"
N_BUILDINGS = 3

CONFIG = """\
format: long
id_column: meter_id
timestamp_column: read_time
load_column: usage
weather_file: weather.csv
weather_temp_column: temp_c
weather_key_column: station_id
"""


def _write_inputs(folder: Path) -> list:
    """Write meter_data.csv, weather.csv, and config.yaml; return building ids."""
    meters, weather, ids = [], [], []

    for path in sorted(SAMPLE_DIR.glob("*.parquet"))[:N_BUILDINGS]:
        bldg_id = int(path.name.split("-")[0])
        frame = pd.read_parquet(path)
        time = frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        meters.append(pd.DataFrame({
            "meter_id": bldg_id, "station_id": f"ST{bldg_id}", "read_time": time,
            "usage": frame["out.electricity.total.energy_consumption..kwh"]}))
        weather.append(pd.DataFrame({
            "station_id": f"ST{bldg_id}", "timestamp": time,
            "temp_c": frame["out.outdoor_air_drybulb_temp..c"]}))
        ids.append(bldg_id)

    folder.mkdir(parents=True, exist_ok=True)
    pd.concat(meters).to_csv(folder / "meter_data.csv", index=False)
    pd.concat(weather).to_csv(folder / "weather.csv", index=False)
    (folder / "config.yaml").write_text(CONFIG)
    return ids


def _reference(ids) -> pd.Series:
    features, profiles, _ = build_inputs(SAMPLE_DIR, InputConfig(format="resstock"), verbose=False)
    return predict(load_bundle(), features, profiles)["hp_probability"].loc[ids]


def test_predict_with_config_from_another_folder(tmp_path, monkeypatch):
    # Run from an unrelated folder: weather_file in the config must be
    # found relative to the config file, not the working directory.
    inputs = tmp_path / "inputs"
    ids = _write_inputs(inputs)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.chdir(run_dir)

    main(["predict", str(inputs / "meter_data.csv"), "--config", str(inputs / "config.yaml"),
          "--out", "predictions.csv"])

    ours = pd.read_csv(run_dir / "predictions.csv").set_index("building_id")
    assert list(ours.index) == sorted(ids)
    np.testing.assert_allclose(ours["hp_probability"].loc[ids], _reference(ids), atol=1e-5)
    assert (ours["quality_flag"] == "ok").all()


def test_threshold_default_and_option(tmp_path, monkeypatch):
    # hp_predicted uses 0.35 by default and follows --threshold when given.
    _write_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    args = ["predict", "meter_data.csv", "--config", "config.yaml"]
    main(args + ["--out", "default.csv"])
    main(args + ["--out", "strict.csv", "--threshold", "0.5"])

    default = pd.read_csv(tmp_path / "default.csv")
    strict = pd.read_csv(tmp_path / "strict.csv")

    assert (default["hp_predicted"] == (default["hp_probability"] >= 0.35)).all()
    assert (strict["hp_predicted"] == (strict["hp_probability"] >= 0.5)).all()
