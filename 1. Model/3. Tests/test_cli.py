"""
Runs the command line on the files in 2. Example/, the way a user would,
and checks the output against 2. Example/predictions.csv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hp_detection.cli import main

EXAMPLES = Path(__file__).resolve().parents[1] / "2. Example"


def test_predict_with_config_from_another_folder(tmp_path, monkeypatch):
    # Run from an unrelated folder: weather_file in the config must be
    # found relative to the config file, not the working directory.
    monkeypatch.chdir(tmp_path)
    main(["predict", str(EXAMPLES / "meter_data.csv"),
          "--config", str(EXAMPLES / "config.yaml"),
          "--out", "predictions.csv"])

    ours = pd.read_csv(tmp_path / "predictions.csv")
    expected = pd.read_csv(EXAMPLES / "predictions.csv")

    assert list(ours["building_id"]) == list(expected["building_id"])
    np.testing.assert_allclose(ours["hp_probability"], expected["hp_probability"], atol=1e-9)
    assert (ours["quality_flag"] == "ok").all()
