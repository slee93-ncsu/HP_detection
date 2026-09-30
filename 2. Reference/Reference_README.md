# Reference: Model Development

This folder documents how the heat pump detection model in `1. Model/` was developed and evaluated. **It is provided for reference only and is not required to apply the model to new data.**

| Path | Contents | Documentation |
|---|---|---|
| `1. Code/` | Feature extraction, evaluation, feature-importance, final-model training, and external validation scripts | `Code_README.md` |
| `2. Results/` | Key evaluation and external validation results | Below |

The original building data is not included in this repository because of its size. Reproducing the results requires downloading it (see [Data](#data)).

---

## Data

The model was developed with ResStock 2025 Release 1 (AMY2018) data for Dallas County, Texas, published by NREL through the Open Energy Data Initiative ([data.openei.org](https://data.openei.org/)).

| Item | Value |
|---|---|
| Buildings | 4,005 in Dallas County (baseline, `upgrade=0`) |
| Period | One calendar year (2018) |
| Interval | 15 minutes (35,040 rows per building), timestamps at interval end, local standard time |
| Timeseries | `https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2025/resstock_amy2018_release_1/timeseries_individual_buildings/by_state/upgrade=0/state=TX/{bldg_id}-0.parquet` |
| Metadata | Texas baseline metadata (`TX_upgrade0`) from the same release |

Only three timeseries columns are used as model inputs:

| Column | Meaning |
|---|---|
| `timestamp` | Interval end |
| `out.electricity.total.energy_consumption..kwh` | Total electricity, kWh per interval |
| `out.outdoor_air_drybulb_temp..c` | Outdoor dry-bulb temperature, °C |

End-use columns (for example, heat pump and backup heating circuits) are not used, since they directly reveal the label.

**Labels.** A building is labeled as a heat pump if `in.hvac_heating_type_and_fuel` in the metadata is `Electricity ASHP` or `Electricity MSHP`. In Dallas County, 1,071 of 4,005 buildings (26.7%) are heat pumps.

---

## Workflow

1. `01_build_features.py` and `02_build_profiles.py` convert each building's timeseries into 102 statistical features and four 24-hour load profiles.
2. `03_train_evaluate.py` evaluates Gradient Boosting, MLP, HybridCNN, and their soft-voting ensemble with stratified 5-fold cross-validation.
3. `04_permutation_importance.py` measures the importance of each input.
4. `05_train_final_model.py` fits the ensemble on all 4,005 buildings and saves the model file used in `1. Model/`.
5. `06_external_validation.py` applies the final model, unchanged, to all 6,063 ResStock buildings in the six counties adjacent to Dallas County (Tarrant, Collin, Denton, Ellis, Kaufman, and Rockwall). None of these buildings were used in training.

Performance is reported from the cross-validation results of step 2, not from the final model fitted on the complete dataset.

---

## Results

| File | Contents |
|---|---|
| `metrics_summary.csv` | Ensemble metrics, mean and standard deviation across the five folds |
| `metrics_per_fold.csv` | The same metrics for each fold |
| `model_comparison.csv` | Metrics of Gradient Boosting, MLP, HybridCNN, and the ensemble |
| `confusion_matrix.csv` | Pooled out-of-fold confusion matrix at threshold 0.5 |
| `misclassification_by_type.csv` | Error rates by heating system type, split into false positives and false negatives |
| `permutation_importance.csv` | ROC-AUC decrease after shuffling each input |
| `group_importance.csv` | Permutation importance aggregated by feature group (A-E) |
| `external_summary.csv` | ROC-AUC, PR-AUC, and metrics at thresholds 0.5 and 0.35 for each adjacent county, all adjacent counties combined, and the Dallas County cross-validation |
| `threshold_comparison.csv` | Precision, recall, and F1-score for each county at thresholds from 0.20 to 0.60 |

**Decision threshold.** In the adjacent counties, ROC-AUC matches the development dataset (0.969 compared with 0.971), but predicted probabilities are lower, and at 0.5 recall drops to 0.585. At 0.35, F1-score is 0.845 in the adjacent counties and 0.848 in the Dallas County cross-validation. The package therefore uses 0.35.

These results are based on simulated buildings and should not be interpreted as expected performance on utility data.
