# Final Trained Model

This folder contains the final trained heat pump detection model generated from the Dallas County development dataset, and the `hp_detection` package that applies it to new meter data.

The package estimates, for each building, the probability that its primary heating system is a heat pump. Only interval electricity consumption and outdoor air temperature are required. Heat pump labels and equipment metadata are not required.

| Path | Contents |
|---|---|
| `hp_detection/` | Input preparation, feature extraction, prediction code, and the model file `hp_detection_model.joblib` |
| `Example/` | Example input files, configuration, and output |
| `tests/` | Checks that predictions reproduce the training pipeline |
| `pyproject.toml` | Package definition; installs the `hp-detect` command |

The tutorial below covers installation through interpretation of results. Reference information on the model follows the tutorial.

---

## Tutorial

### Step 1. Install the package

Python 3.11 or later is required. A new virtual environment is recommended.

```bash
cd "1. Model"
pip install .
```

`scikit-learn` is pinned to 1.8.0 because the model file stores scikit-learn 1.8.0 objects.

### Step 2. Run the example

```bash
hp-detect predict Example/meter_data.csv --config Example/config.yaml --out predictions.csv
```

The command scores four ResStock buildings in Tarrant County, Texas, which were not used in training. A successful run ends with:

```
Buildings scored     : 4
Predicted heat pump  : 2 (50.0%)
Low data quality     : 0
```

`predictions.csv` should match `Example/predictions.csv`.

### Step 3. Prepare the meter data

Place all buildings in one CSV or parquet table, one row per building per reading.

| Column | Example | Unit |
|---|---|---|
| Meter ID | `A1001` | - |
| Time | `2023-01-01 00:15` | local time |
| Electricity | `0.42` | kWh per interval |

```
meter_id,read_time,usage
A1001,2023-01-01 00:15,0.42
A1001,2023-01-01 00:30,0.38
```

Requirements:

- One full year per building.
- Readings at intervals of one hour or shorter (15, 30, or 60 minutes). The interval is detected per building.
- Total electricity consumption, not net of rooftop solar.
- Missing readings left blank or omitted, not recorded as 0.

Column names may differ from the example; they are specified in Step 5.

### Step 4. Prepare the outdoor temperature

| Column | Example | Unit |
|---|---|---|
| Time | `2023-01-01 00:15` | local time |
| Outdoor dry-bulb temperature | `5.1` | °C |

```
timestamp,temp_c
2023-01-01 00:15,5.1
2023-01-01 00:30,4.7
```

The temperature series must cover the same year as the meter data. If temperature is already a column in the meter data, this file is not needed.

### Step 5. Write the configuration file

The configuration file maps the columns of Steps 3 and 4 to their roles. `Example/config.yaml`:

```yaml
format: long
id_column: meter_id
timestamp_column: read_time
load_column: usage
weather_file: weather.csv
weather_temp_column: temp_c
```

If temperature is in the meter data, replace the two `weather_` lines with `temp_column: <column name>`. A relative `weather_file` path is resolved from the configuration file's folder.

The defaults match the training data: kWh per interval, °C, and timestamps marking the end of each interval. For other formats, add the corresponding setting:

| Data format | Setting |
|---|---|
| Wh per interval | `load_unit: Wh` |
| kW average demand | `load_unit: kW` |
| °F | `temp_unit: F` |
| Timestamps at the start of each interval (e.g. `00:00` for 00:00-00:15) | `timestamp_convention: start` |
| Timestamps with a UTC offset (e.g. `-06:00`) | `timezone: America/Chicago` |
| Several weather stations | `weather_key_column: station_id` (present in both files) |

All settings are listed with comments in `Example/config_template.yaml`.

### Step 6. Run the prediction

```bash
hp-detect predict meter_data.csv --config config.yaml --out predictions.csv
```

The equivalent Python call:

```python
from hp_detection import InputConfig, run

cfg = InputConfig(id_column="meter_id", timestamp_column="read_time",
                  load_column="usage", weather_file="weather.csv",
                  weather_temp_column="temp_c")
result = run("meter_data.csv", cfg)
```

Readings are converted to the training format, summed to hourly values, and passed through the same feature extraction as in training. `--workers N` processes buildings in parallel.

### Step 7. Review the output

`predictions.csv` contains one row per building. Output of Step 2:

| building_id | hp_probability | hp_predicted | annual_kwh | interval_minutes | quality_flag |
|---|---:|---:|---:|---:|---|
| M352201 | 0.614 | 1 | 13958 | 15 | ok |
| M380181 | 0.608 | 1 | 39492 | 15 | ok |
| M498044 | 0.004 | 0 | 16089 | 15 | ok |
| M498133 | 0.385 | 0 | 25475 | 15 | ok |

| Column | Contents |
|---|---|
| `hp_probability` | Soft-voting probability of a heat pump (0-1) |
| `hp_predicted` | 1 if `hp_probability` is at least the decision threshold (0.5) |
| `p_gradient_boosting`, `p_mlp`, `p_cnn` | Base model probabilities |
| `annual_kwh` | Total electricity over the data period |
| `interval_minutes` | Detected reading interval |
| `span_days`, `load_coverage_pct`, `temp_coverage_pct` | Data period and share of hours with data |
| `quality_flag` | `ok`, or `low` if the input differs from a full, gap-free year |
| `quality_notes` | `less_than_one_year`, `missing_months`, `load_gaps`, `temperature_gaps`, `negative_load`; `duplicate_timestamps` is informational |

Predictions cannot be scored without known heat pump status. The following checks are recommended:

- Buildings flagged `low` should be treated with caution.
- `interval_minutes` should match the known reading interval.
- `annual_kwh` values several times higher or lower than expected for the service area typically indicate a unit setting error (for example, a factor of 4 when 15-minute kW data is read as kWh).
- The overall predicted heat pump share can be compared with published regional statistics.
- `hp_probability` is suited to ranking buildings; `hp_predicted` applies a fixed threshold.

`--save-inputs DIR` also writes the computed features and load profiles.

---

## Model File

`hp_detection/hp_detection_model.joblib`

This file is created by `05_train_final_model.py` after the evaluation workflow is complete.

Unlike the models used during 5-fold cross-validation, this final model is trained using all buildings that are present in the feature, profile, and metadata inputs.

`hp-detect info` prints the training metadata stored in the file.

## Model Contents

The model bundle contains:

- fitted Gradient Boosting classifier,
- fitted MLP classifier,
- fitted StandardScaler,
- learned HybridCNN parameters,
- statistical feature ordering,
- load-profile ordering,
- decision threshold,
- positive heat-pump label definitions, and
- basic training and software metadata.

The final heat-pump probability is produced by averaging the prediction probabilities from Gradient Boosting, MLP, and HybridCNN.

## Training Data

The model was trained using the Dallas County ResStock 2025 Release 1 development dataset.

The final training set includes only buildings that are present in all three required inputs:

- statistical features generated by `01_build_features.py`,
- load profiles generated by `02_build_profiles.py`, and
- heating-system metadata.

The heating-system metadata is used to define the training labels. The model inputs themselves are based on building-level electricity consumption and outdoor air temperature.

## Reproducibility

The final model can be regenerated using:

`2. Model_Development/2. Code/05_train_final_model.py`

with `--model-out "1. Model/hp_detection/hp_detection_model.joblib"` so the package uses the new model.

The feature and profile inputs are generated by:

- `2. Model_Development/2. Code/01_build_features.py`
- `2. Model_Development/2. Code/02_build_profiles.py`

Evaluation is performed separately using:

- `2. Model_Development/2. Code/03_train_evaluate.py`

## Tests

```bash
pip install ".[test]"
pytest
```

The tests confirm that features, load profiles, and predictions computed by the package match the training pipeline on sample buildings, that the same data supplied in other units and layouts gives the same predictions, and that the command line runs on `Example/`.

## Notes

Performance on other datasets may differ because of differences in climate, building stock, heat-pump prevalence, data resolution, missing data, and other dataset characteristics. The model was trained on simulated buildings in a single county.

Other electric heating systems (electric furnace, baseboard, boiler) are the most common source of false positives. Error rates by heating type are listed in `2. Model_Development/3. Output/2. Results/misclassification_by_type.csv`.

Only load serialized model files such as `joblib` from trusted sources.
