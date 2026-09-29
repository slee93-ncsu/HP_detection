"""
Evaluate the final model on ResStock buildings outside the training county.

The final model (1. Model/) is applied, unchanged, to every ResStock 2025
Release 1 building in the counties listed below. These buildings were not
used in training. Heating-system metadata provides the true labels.

Inputs:
    metadata CSV (or the zip containing it)   bldg_id, in.county_name,
                                              in.hvac_heating_type_and_fuel, ...
    building timeseries                       downloaded from the Open Energy
                                              Data Initiative on first run and
                                              cached in --data-dir (three columns)
    03 predictions.csv (optional)             adds Dallas County cross-validation
                                              as a reference row

Outputs, written to --work-dir:
    external_predictions.csv    per building: county, heating type, building
                                type, true label, model probabilities
    external_summary.csv        per county: ROC-AUC, PR-AUC, and thresholded
                                metrics at 0.5 and 0.35
    threshold_comparison.csv    per county and threshold: precision, recall, F1

Requires the hp_detection package (pip install "1. Model").

Usage:
    python 06_external_validation.py --metadata "path/to/TX_upgrade0_Residential.zip" \
        --reference-predictions "../3. Output/2. Results/predictions.csv" --workers 8
"""

from __future__ import annotations

import argparse
import io
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from hp_detection import InputConfig, build_inputs, load_bundle, predict

COUNTIES = ["Tarrant County", "Collin County", "Denton County",
            "Ellis County", "Kaufman County", "Rockwall County"]

HP_HEATING_TYPES = {"Electricity ASHP", "Electricity MSHP"}
REPORT_THRESHOLDS = (0.5, 0.35)
THRESHOLD_GRID = np.round(np.arange(0.20, 0.61, 0.05), 2)

TIMESERIES_URL = (
    "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/"
    "end-use-load-profiles-for-us-building-stock/2025/resstock_amy2018_release_1/"
    "timeseries_individual_buildings/by_state/upgrade=0/state=TX/{}-0.parquet"
)
COLUMNS = ["timestamp", "out.electricity.total.energy_consumption..kwh",
           "out.outdoor_air_drybulb_temp..c"]
META_COLUMNS = ["bldg_id", "in.county_name", "in.hvac_heating_type_and_fuel",
                "in.geometry_building_type_recs", "in.weather_file_city"]


def download(bldg_id: int, dst: Path) -> bool:
    """Save the three model columns of one building. Returns False on failure."""
    if dst.exists():
        return True

    for attempt in range(5):
        try:
            data = urllib.request.urlopen(TIMESERIES_URL.format(bldg_id), timeout=180).read()
            pd.read_parquet(io.BytesIO(data), columns=COLUMNS).to_parquet(dst, index=False)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(2 ** attempt)

    return False


def metrics(y, prob, threshold) -> dict:
    pred = (prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "predicted_hp_share": pred.mean(),
    }


def summarize(name, y, prob, weather="") -> dict:
    row = {"county": name, "n": len(y), "hp_share": y.mean(), "weather_station": weather,
           "roc_auc": roc_auc_score(y, prob), "pr_auc": average_precision_score(y, prob),
           "mean_prob_true_hp": prob[y == 1].mean()}
    for t in REPORT_THRESHOLDS:
        row.update({f"{k}_{t}": v for k, v in metrics(y, prob, t).items()})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--data-dir", default="external_data")
    parser.add_argument("--work-dir", default="outputs")
    parser.add_argument("--reference-predictions", default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    data_dir, work_dir = Path(args.data_dir), Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.metadata, usecols=META_COLUMNS, low_memory=False)
    meta = meta[meta["in.county_name"].isin(COUNTIES)].copy()
    meta["true_label"] = meta["in.hvac_heating_type_and_fuel"].isin(HP_HEATING_TYPES).astype(int)
    print(f"Buildings in {len(COUNTIES)} counties: {len(meta):,}")

    bundle = load_bundle()
    results = []

    for county, rows in meta.groupby("in.county_name"):
        folder = data_dir / county.replace(" ", "_")
        folder.mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(16) as executor:
            ok = list(executor.map(lambda b: download(b, folder / f"{b}-0.parquet"), rows["bldg_id"]))
        print(f"  {county}: {sum(ok):,}/{len(ok):,} buildings available")

        features, profiles, _ = build_inputs(folder, InputConfig(format="resstock"),
                                             args.workers, verbose=False)
        scores = predict(bundle, features, profiles)
        results.append(rows.set_index("bldg_id").join(scores, how="inner"))

    predictions = pd.concat(results)
    predictions.index.name = "bldg_id"
    predictions.to_csv(work_dir / "external_predictions.csv")

    summary, comparison = [], []
    groups = [(c, g) for c, g in predictions.groupby("in.county_name")]
    groups.append(("All external counties", predictions))

    if args.reference_predictions:
        ref = pd.read_csv(args.reference_predictions)
        ref = pd.DataFrame({"true_label": ref["true_label"], "hp_probability": ref["ensemble_prob"],
                            "in.weather_file_city": "Dallas Love Fld"})
        groups.insert(0, ("Dallas County (5-fold CV)", ref))

    for name, g in groups:
        y, prob = g["true_label"].to_numpy(), g["hp_probability"].to_numpy()
        weather = g["in.weather_file_city"].mode()[0] if name != "All external counties" else ""
        summary.append(summarize(name, y, prob, weather))
        for t in THRESHOLD_GRID:
            comparison.append({"county": name, "threshold": t, **metrics(y, prob, t)})

    pd.DataFrame(summary).round(4).to_csv(work_dir / "external_summary.csv", index=False)
    pd.DataFrame(comparison).round(4).to_csv(work_dir / "threshold_comparison.csv", index=False)

    print(pd.DataFrame(summary)[["county", "n", "roc_auc", "f1_0.5", "f1_0.35"]].round(3).to_string(index=False))
    print(f"\nWrote results to {work_dir}")


if __name__ == "__main__":
    main()
