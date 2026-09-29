# HP_detection

Heat pump detection from building-level electricity consumption and outdoor air temperature.

Given a year of interval meter data and a co-located temperature series, the model estimates the probability that each household has a heat pump. Heat pump labels and equipment metadata are not required. The model was developed and evaluated using ResStock 2025 Release 1 data for Dallas County, Texas.

---

## Repository Layout

| Path | Contents |
|---|---|
| `1. Model/` | Final trained model and the `hp_detection` package that applies it to new meter data |
| `2. Model_Development/` | Data, scripts, and evaluation results used to develop the model (reference) |

For installation, input requirements, and outputs, see [`1. Model/Model_README.md`](1.%20Model/Model_README.md).

For the development workflow, see [`2. Model_Development/Development_README.md`](2.%20Model_Development/Development_README.md).

---

## Using the Model

Python 3.11 or later is required.

```bash
cd "1. Model"
pip install .
hp-detect predict Example/meter_data.csv --config Example/config.yaml --out predictions.csv
```

The command above runs the example in `1. Model/Example/`. For new data, the meter data and outdoor temperature files are described in a configuration file; see [`1. Model/Model_README.md`](1.%20Model/Model_README.md).

The output contains, for each building:

| Column | Contents |
|---|---|
| `hp_probability` | Probability of a heat pump (0-1) |
| `hp_predicted` | 1 if `hp_probability` is at least 0.5 |
| `quality_flag` | `ok`, or `low` if the input differs from a full, gap-free year |

---

## Model

Each building's data is resampled to hourly values and converted into 102 statistical features and four 24-hour load profiles. Gradient Boosting, MLP, and HybridCNN each produce a heat pump probability, and the three are averaged (soft voting).

A building is labeled as a heat pump if its primary heating system is an air-source heat pump (`Electricity ASHP`) or a ductless mini-split (`Electricity MSHP`). Only electricity consumption and outdoor temperature are used as model inputs. Heating-system metadata is used to define the training and evaluation labels.

The final model is trained on the complete Dallas County development dataset (4,005 buildings) and stored as `1. Model/hp_detection/hp_detection_model.joblib`.

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

Detailed evaluation outputs are provided under `2. Model_Development/3. Output/`.

---

## Requirements

Python 3.11 or later with:

- numpy
- pandas
- pyarrow
- scikit-learn
- torch
- joblib

`pip install .` in `1. Model/` installs the required packages. Pinned versions for the development scripts are provided in [`2. Model_Development/2. Code/requirements.txt`](2.%20Model_Development/2.%20Code/requirements.txt).

The model runs on CPU. The development scripts can run on CPU or GPU.
