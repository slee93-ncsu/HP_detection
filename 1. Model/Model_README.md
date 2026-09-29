# Final Trained Model

This folder contains the final trained heat pump detection model generated from the Dallas County development dataset, and the `hp_detection` package that applies it to new meter data.

The package estimates, for each building, the probability that its primary heating system is a heat pump. Only interval electricity consumption and outdoor air temperature are required. Heat pump labels and equipment metadata are not required.

## Folder Layout

| Path | Contents |
|---|---|
| `hp_detection/` | Input preparation, feature extraction, prediction code, and the model file `hp_detection_model.joblib` |
| `Example/` | Example input files, configuration, and output |
| `tests/` | Checks that predictions reproduce the training pipeline |
| `pyproject.toml` | Package definition; installs the `hp-detect` command |

## Installation

Python 3.11 or later is required. A new virtual environment is recommended.

```bash
cd "1. Model"
pip install .
```

`scikit-learn` is pinned to 1.8.0 because the model file stores scikit-learn 1.8.0 objects.

## Usage

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

`hp-detect info` prints the training metadata stored in the model file.

## Input Requirements

Two files are required: meter data and outdoor temperature. Both may be CSV or parquet. Column names are specified in a configuration file.

### Meter data

All buildings in one table, one row per building per reading.

| Column | Example | Unit |
|---|---|---|
| Meter ID | `A1001` | - |
| Time | `2023-01-01 00:15` | local time |
| Electricity | `0.42` | kWh per interval |

### Outdoor temperature

| Column | Example | Unit |
|---|---|---|
| Time | `2023-01-01 00:15` | local time |
| Outdoor dry-bulb temperature | `5.1` | °C |

If temperature is included in the meter data, the separate file is not needed (`temp_column` replaces `weather_file`).

### Data requirements

- One full year per building.
- Readings at intervals of one hour or shorter (15, 30, or 60 minutes). The interval is detected per building.
- Total electricity consumption, not net of rooftop solar.
- Missing readings left blank or omitted, not recorded as 0.

### Configuration file

The configuration file maps columns to their roles. `Example/config.yaml`:

```yaml
format: long
id_column: meter_id
timestamp_column: read_time
load_column: usage
weather_file: weather.csv
weather_temp_column: temp_c
```

The defaults match the training data: kWh per interval, °C, and timestamps marking the end of each interval. Other formats are converted before any feature is computed:

| Data format | Setting |
|---|---|
| Wh per interval | `load_unit: Wh` |
| kW average demand | `load_unit: kW` |
| °F | `temp_unit: F` |
| Timestamps at the start of each interval | `timestamp_convention: start` |
| Timestamps with a UTC offset (e.g. `-06:00`) | `timezone: America/Chicago` |
| Several weather stations | `weather_key_column: station_id` (present in both files) |

A relative `weather_file` path is resolved from the configuration file's folder. All settings are listed with comments in `Example/config_template.yaml`.

Readings are summed to hourly values before feature extraction, as in training. Timestamps with a UTC offset are converted to local standard time for the full year, matching the training data. If a building's reading interval changes within the year, its data should be supplied in kWh with interval-end timestamps.

## Outputs

`hp-detect predict` writes one row per building.

| Column | Contents |
|---|---|
| `building_id` | Meter ID from the input |
| `hp_probability` | Soft-voting probability of a heat pump (0-1) |
| `hp_predicted` | 1 if `hp_probability` is at least the decision threshold (0.5) |
| `p_gradient_boosting`, `p_mlp`, `p_cnn` | Base model probabilities |
| `annual_kwh` | Total electricity over the data period |
| `interval_minutes` | Detected reading interval |
| `span_days`, `load_coverage_pct`, `temp_coverage_pct` | Data period and share of hours with data |
| `quality_flag` | `ok`, or `low` if the input differs from a full, gap-free year |
| `quality_notes` | `less_than_one_year`, `missing_months`, `load_gaps`, `temperature_gaps`, `negative_load`; `duplicate_timestamps` is informational |

`--save-inputs DIR` also writes the computed features and load profiles.

Predictions cannot be scored without known heat pump status. Buildings flagged `low` should be treated with caution, and the overall predicted heat pump share can be compared with published regional statistics. `annual_kwh` values far outside 5,000-20,000 kWh typically indicate a unit setting error.

## Example

`Example/` contains one year of 15-minute data for four ResStock buildings in Tarrant County, Texas. These buildings were not used in training.

```bash
hp-detect predict Example/meter_data.csv --config Example/config.yaml --out predictions.csv
```

| Building | Heating system | `hp_probability` | `hp_predicted` |
|---|---|---:|---:|
| M352201 | Electricity ASHP | 0.61 | 1 |
| M380181 | Electricity ASHP | 0.61 | 1 |
| M498044 | Natural Gas Fuel Furnace | 0.00 | 0 |
| M498133 | Electricity Electric Furnace | 0.39 | 0 |

The example illustrates the input and output format and is not a measure of accuracy.

## Model File

`hp_detection/hp_detection_model.joblib`

This file is created by `05_train_final_model.py` after the evaluation workflow is complete.

Unlike the models used during 5-fold cross-validation, this final model is trained using all buildings that are present in the feature, profile, and metadata inputs.

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
