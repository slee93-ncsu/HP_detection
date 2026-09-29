# Model Development

How the model was built. You do not need this folder to use the model.

| Folder | What's inside |
|---|---|
| `data/` | 10 sample homes and the home information file (heating type) |
| `scripts/` | The 5 scripts that built the model |
| `results/` | Test results and the values used for training |

## Steps

1. **Data.** NREL ResStock 2025, 4,005 simulated homes in Dallas County,
   Texas. Each home has one year of electricity use and outdoor
   temperature every 15 minutes. Only these two are used.
2. **Labels.** A home is a heat pump home if its heating type is
   `Electricity ASHP` or `Electricity MSHP`. 1,071 homes (27%).
3. **Prepare.** Combine readings into hourly values. Convert temperature
   to °F.
4. **Features** (`01_build_features.py`, `02_build_profiles.py`).
   102 summary numbers per home in five groups, plus four 24-hour usage
   curves:

   | Group | Describes |
   |---|---|
   | A | Yearly usage level and spread |
   | B | Seasonal and monthly usage |
   | C | Time-of-day usage |
   | D | Usage in cold, mild, warm, and hot weather |
   | E | How fast usage rises as temperature changes |

   Usage curves: winter weekday, summer weekday, spring/fall weekday, and
   the highest-use day.
5. **Models** (`03_train_evaluate.py`). Three models, averaged:
   Gradient Boosting, a small neural network (MLP), and a convolutional
   network that reads the usage curves (HybridCNN).
6. **Testing** (`03_train_evaluate.py`, `04_permutation_importance.py`).
   5-fold cross-validation, and a check of which inputs matter most.
7. **Final model** (`05_train_final_model.py`). Retrained on all 4,005
   homes and saved as `hp_detection/hp_detection_model.joblib`.

Results are summarized in the [model card](../docs/model_card.md).

## Rerun

Install the versions in `scripts/requirements.txt` and unzip
`data/TX_upgrade0_Residential.zip`. Then, from the repository root:

```bash
python model_development/scripts/01_build_features.py --parquet-dir path/to/timeseries
python model_development/scripts/02_build_profiles.py --parquet-dir path/to/timeseries
python model_development/scripts/03_train_evaluate.py --metadata path/to/TX_upgrade0_Residential.csv
python model_development/scripts/04_permutation_importance.py --metadata path/to/TX_upgrade0_Residential.csv
python model_development/scripts/05_train_final_model.py --metadata path/to/TX_upgrade0_Residential.csv --model-out hp_detection/hp_detection_model.joblib
```

The full data (all 4,005 homes) is not included; it is available from the
[Open Energy Data Initiative](https://data.openei.org/). Keep the five
scripts in one folder.

## Results files

| Folder | Files |
|---|---|
| `results/model_inputs/` | The 102 numbers and usage curves for every home |
| `results/evaluation/` | Accuracy (`metrics_summary.csv`), confusion matrix, errors by heating type, per-home predictions, and more |
| `results/feature_importance/` | Which inputs matter most |
