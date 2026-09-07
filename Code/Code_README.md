# Heat Pump Detection from Building Load Profiles

Classifies whether a dwelling has a heat pump using two signals only:
**building-level electricity consumption** and **outdoor air temperature**.

Four scripts: feature extraction, profile extraction, training and
evaluation, and permutation importance.


---

## Pipeline

| Script | Reads | Writes |
|---|---|---|
| `01_build_features.py` | Timeseries parquet | `dallas_features_groupA_E.csv` — 102 features per building |
| `02_build_profiles.py` | Timeseries parquet | `dallas_cnn_profiles.csv` — 4 × 24 load profiles per building |
| `03_train_evaluate.py` | Outputs of 01 and 02, plus metadata | Result tables and out-of-fold predictions |
| `04_permutation_importance.py` | Same as 03 | ROC-AUC drop per input |

Run in order. All four must sit in the same folder; script 04 loads
script 03 by file path.


---

## Input requirements

### Timeseries files

One file per building named `{bldg_id}-{upgrade}.parquet`, in the
directory given by `--parquet-dir`. Three columns are read:

| Column | Meaning |
|---|---|
| `timestamp` | Datetime |
| `out.electricity.total.energy_consumption..kwh` | Total electricity, kWh per interval |
| `out.outdoor_air_drybulb_temp..c` | Outdoor drybulb temperature, °C |

Column names and the filename pattern are constants at the top of
scripts 01 and 02; change them there for a different schema.

Any interval at or below one hour works. The scripts resample to hourly,
summing load and averaging temperature. Seasonal and monthly features
assume a full year of data.

Temperature is converted to Fahrenheit internally and all thresholds are
defined in Fahrenheit. Supply Celsius, or adjust `to_fahrenheit()`.

### Metadata

A CSV with one row per building, given by `--metadata`. Two columns are
used: `bldg_id` and `in.hvac_heating_type_and_fuel`.

A building is positive if its heating type is in `HP_HEATING_TYPES`,
defined at the top of script 03 as `{"Electricity ASHP",
"Electricity MSHP"}`. Edit that set to match your own equipment labels.

The three inputs are inner-joined on `bldg_id`, so the metadata may
cover a wider population than the timeseries.

---

## Method

**Features.** 102 statistical features per building.

| Group | n | Content |
|---|---|---|
| A | 25 | Annual load statistics, percentiles, dispersion |
| B | 22 | Season means and standard deviations, season ratios, monthly means |
| C | 8 | Time-of-day load ratios, peak and trough hour |
| D | 37 | Per-temperature-range statistics and cross-range ratios |
| E | 10 | Load–temperature slopes over several ranges, and their contrasts |

Temperature ranges: below 40 °F, 40–60 °F, 60–80 °F, and 80 °F and
above. Group E fits slopes on 5 °F binned means so each band contributes
equally regardless of how many hours fall in it.

**Profiles.** Four 24-hour mean load curves per building: weekday
winter, weekday summer, weekday shoulder, and the single
highest-consumption day. Columns are profile-major and reshaped to
(n, 4, 24) in script 03.

Script 01 uses a four-season calendar map, script 02 a three-season map.
They are independent.

**Models.** Three learners are trained per fold and averaged.

| Model | Input |
|---|---|
| GradientBoosting | Raw 102 features |
| MLP (64, 32) | Standardized 102 features |
| HybridCNN | (4, 24) profile tensor + standardized 102 features |
| SoftVoting | Mean of the three probabilities |
| Stacking | Optional, `--with-stacking`. Logistic meta-learner on out-of-fold probabilities; about 6× the runtime. |

**Evaluation.** Stratified 5-fold cross-validation. All numbers come
from out-of-fold predictions. The scaler is fitted inside the fold loop
on training rows only. Cross-validation runs once and every table is
derived from that single set of predictions.

Hyperparameters, seed, fold count, and decision threshold are constants
at the top of script 03.

---

## Outputs

From script 03:

| File | Contents |
|---|---|
| `predictions.csv` | Per building: each model's probability, predicted and true label, fold |
| `metrics_summary.csv` | Ensemble metrics, mean and standard deviation across folds |
| `metrics_per_fold.csv` | The same metrics per fold |
| `model_comparison.csv` | Metrics for each model |
| `model_ablation.csv` | Metrics for all subsets of the three base models |
| `confusion_matrix.csv` | Pooled out-of-fold confusion matrix |
| `model_correlation.csv` | Pairwise correlation of base model probabilities |
| `error_overlap.csv` | Buildings grouped by how many base models misclassify them |
| `misclassification_by_type.csv` | Error rates by true heating system, split into false positives and negatives |
| `run_summary.json` | Platform, library versions, seed, sample size |

From script 04: `permutation_importance.csv` (ROC-AUC drop per input,
sorted) and `group_importance.csv` (aggregated to groups A–E).

---

## Notes on results

Read accuracy together with ROC-AUC, PR-AUC, and the class-specific
error rates. In an imbalanced sample, predicting the majority class
everywhere already scores well on accuracy.

`misclassification_by_type.csv` shows which equipment categories the
model handles well. Categories with few buildings will have unstable
rates.

The confusion matrix is evaluated at a 0.5 probability threshold and is
the least stable output across runs, since buildings near that boundary
move between cells. ROC-AUC is unaffected. The CNN is a PyTorch model,
so results vary slightly across hardware and library versions even with
a fixed seed; report the fold-level standard deviation with the mean.

---

## Environment

Requires numpy, pandas, pyarrow, scikit-learn, and torch. Pinned
versions in `requirements.txt`. Runs on CPU or GPU.
