"""
End-to-end inference: raw meter data -> features -> predictions.

    from hp_detection import InputConfig, run
    result = run("meter_data.csv", InputConfig(id_column="meter_id", ...))
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import pandas as pd

from . import io, quality
from .features import compute_features
from .model import load_bundle, predict
from .profiles import compute_profiles


def _process(item) -> Optional[tuple]:
    """Compute features, profiles and quality for one building.
    `item` is (id, hourly, raw_info) or (id, resstock_path)."""
    building_id = item[0]
    try:
        if len(item) == 2:
            hourly, raw_info = io.read_resstock_file(item[1])
        else:
            hourly, raw_info = item[1], item[2]

        return (building_id,
                compute_features(hourly),
                compute_profiles(hourly),
                quality.assess(hourly, raw_info))
    except Exception as error:  # noqa: BLE001
        print(f"  FAILED {building_id}: {error}", file=sys.stderr)
        return None


def build_inputs(input_path, cfg: io.InputConfig, workers: int = 1, verbose: bool = True):
    """Return (features, profiles, quality) DataFrames indexed by building id."""
    cfg.validate()
    fmt = io.detect_format(input_path) if cfg.format == "auto" else cfg.format

    if fmt == "resstock":
        items = io.iter_resstock_dir(input_path)
    elif fmt == "long":
        items = io.iter_long_table(input_path, cfg)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_process, items, chunksize=16))
    else:
        results = [_process(item) for item in items]

    ok = sorted((r for r in results if r is not None), key=lambda r: r[0])
    if verbose:
        print(f"Buildings processed: {len(ok):,}  failed: {len(results) - len(ok):,}")
    if not ok:
        raise SystemExit("No buildings processed successfully.")

    ids = pd.Index([r[0] for r in ok], name="building_id")
    features = pd.DataFrame([r[1] for r in ok], index=ids)
    profiles = pd.DataFrame([r[2] for r in ok], index=ids)
    checks = pd.DataFrame([r[3] for r in ok], index=ids)

    return features, profiles, checks


def run(input_path, cfg: Optional[io.InputConfig] = None, model_path=None,
        workers: int = 1, verbose: bool = True,
        threshold: Optional[float] = None) -> pd.DataFrame:
    """Predict heat-pump presence for every building in `input_path`.
    `threshold` defaults to model.DEFAULT_THRESHOLD (0.35)."""
    cfg = cfg or io.InputConfig()
    bundle = load_bundle(model_path)

    features, profiles, checks = build_inputs(input_path, cfg, workers, verbose)
    scores = predict(bundle, features, profiles, threshold)

    return scores.join(checks)
