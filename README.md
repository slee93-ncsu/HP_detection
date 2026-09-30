# HP_detection

Heat pump detection from building-level electricity consumption and outdoor air temperature.

Given a year of interval meter data and a co-located temperature series, the model estimates the probability that each household has a heat pump. Heat pump labels and equipment metadata are not required. The model was developed and evaluated using ResStock 2025 Release 1 data for Dallas County, Texas.

---

## Repository Layout

| Path | Contents |
|---|---|
| `1. Model/` | Final trained model and the `hp_detection` package that applies it to new meter data |
| `2. Reference/` | Scripts and key results from the development of the model. For reference only; not required to use the model |

For installation, input requirements, and outputs, see [`1. Model/Model_README.md`](1.%20Model/Model_README.md).

For how the model was developed, see [`2. Reference/Reference_README.md`](2.%20Reference/Reference_README.md).

---

## Using the Model

Python 3.11 or later is required.

```bash
cd "1. Model"
pip install .
hp-detect predict meter_data.csv --config config.yaml --out predictions.csv
```

The meter data and outdoor temperature files are described in a configuration file. File formats, settings, and a step-by-step tutorial are in [`1. Model/Model_README.md`](1.%20Model/Model_README.md).

The output contains, for each building:

| Column | Contents |
|---|---|
| `hp_probability` | Probability of a heat pump (0-1) |
| `hp_predicted` | 1 = likely heat pump, 0 = likely not (1 when `hp_probability` is 0.35 or higher) |
| `quality_flag` | `ok`, or `low` if the data is incomplete and the result is less reliable |

---

## Model

Each building's data is resampled to hourly values and converted into 102 statistical features and four 24-hour load profiles. Gradient Boosting, MLP, and HybridCNN each produce a heat pump probability, and the three are averaged (soft voting).

A building is labeled as a heat pump if its primary heating system is an air-source heat pump (`Electricity ASHP`) or a ductless mini-split (`Electricity MSHP`). Only electricity consumption and outdoor temperature are used as model inputs. Heating-system metadata is used to define the training and evaluation labels.

The final model is trained on the complete Dallas County development dataset (4,005 buildings) and stored as `1. Model/1. Package/hp_detection/hp_detection_model.joblib`.

---

## Reference Performance

The soft-voting ensemble achieved the following mean performance across 5-fold cross-validation on the Dallas County development dataset:

| Metric | Mean | Std |
|---|---:|---:|
| Accuracy | 0.918 | 0.012 |
| Precision | 0.859 | 0.018 |
| Recall | 0.828 | 0.045 |
| F1-score | 0.843 | 0.025 |
| ROC-AUC | 0.971 | 0.008 |
| PR-AUC | 0.926 | 0.018 |

These values are provided as reference performance on the development dataset and should not be interpreted as expected performance on utility data. Performance may vary with climate, building stock, heat pump prevalence, data resolution, missing data, and other dataset characteristics.

Detailed evaluation results are provided under `2. Reference/2. Results/`.

---

## External Validation

The final model was applied, unchanged, to all 6,063 ResStock buildings in the six counties adjacent to Dallas County. None of these buildings were used in training.

| County | Buildings | ROC-AUC | F1-score at 0.5 | F1-score at 0.35 |
|---|---:|---:|---:|---:|
| Dallas (5-fold CV) | 4,005 | 0.971 | 0.843 | 0.848 |
| Tarrant | 3,037 | 0.969 | 0.775 | 0.861 |
| Collin | 1,351 | 0.966 | 0.554 | 0.784 |
| Denton | 1,153 | 0.963 | 0.596 | 0.828 |
| Ellis | 235 | 0.992 | 0.850 | 0.926 |
| Kaufman | 161 | 0.998 | 0.677 | 0.892 |
| Rockwall | 126 | 0.975 | 0.667 | 0.853 |
| All adjacent counties | 6,063 | 0.969 | 0.709 | 0.845 |

ROC-AUC in the adjacent counties matches the development dataset, so the model ranks buildings equally well. Predicted probabilities are, however, lower outside Dallas County, and at 0.5 many heat pumps are missed (recall 0.40 in Collin County). A threshold of 0.35 restores performance in every adjacent county without reducing performance in Dallas County, and is therefore used by the package. The threshold can be changed with `--threshold`.

These results are based on simulated buildings. Validation against a sample of utility customers with known heating equipment (for example, rebate or audit records) is recommended before operational use.

Detailed results are provided in `2. Reference/2. Results/external_summary.csv` and `threshold_comparison.csv`.

---

## Requirements

Python 3.11 or later with:

- numpy
- pandas
- pyarrow
- scikit-learn
- torch
- joblib

`pip install .` in `1. Model/` installs the required packages. Pinned versions for the development scripts are provided in [`2. Reference/1. Code/requirements.txt`](2.%20Reference/1.%20Code/requirements.txt).

The model runs on CPU. The development scripts can run on CPU or GPU.
