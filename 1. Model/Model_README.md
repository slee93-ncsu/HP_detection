# Final Trained Model

This folder contains the final trained heat pump detection model generated from the Dallas County development dataset, and the `hp_detection` package that applies it to new meter data.

The package estimates, for each building, the probability that its primary heating system is a heat pump. **Only interval electricity consumption and outdoor air temperature are required.** Heat pump labels and equipment metadata are not required.

| Path | Contents |
|---|---|
| `1. Package/` | The `hp_detection` package: input preparation, feature extraction, prediction code, and the model file `hp_detection_model.joblib` |
| `2. Tests/` | Checks that predictions reproduce the training pipeline (`pip install ".[test]"`, then `pytest`) |
| `pyproject.toml` | Package definition; installs the `hp-detect` command |

The tutorial below covers installation through interpretation of results. How the model was built is documented in `2. Model_Development/`.

---

## Tutorial

### Step 1. Install the package

Installation is done once per computer. An internet connection is required.

**1. Download the repository.** On the GitHub page of this repository, select **Code** → **Download ZIP**, and unzip the file. (With Git installed, `git clone` may be used instead.)

**2. Install Python 3.11 or later** from [python.org](https://www.python.org/downloads/). On Windows, check **Add python.exe to PATH** during installation. To confirm, open a terminal and run `python --version`.

**3. Open a terminal in the `1. Model` folder.** On Windows, open the `1. Model` folder in File Explorer, type `cmd` in the address bar, and press Enter. A Command Prompt opens in that folder.

**4. Create and activate a virtual environment.** A virtual environment keeps the packages for this model separate from other Python software on the computer.

```bash
python -m venv .venv
.venv\Scripts\activate
```

On Mac or Linux, the second line is `source .venv/bin/activate`. When active, `(.venv)` appears at the start of the terminal line.

**5. Install the package.**

```bash
pip install .
```

This installs the model and the libraries it needs, which may take several minutes. PyTorch is the largest: several hundred MB on Windows and Mac, and several GB on Linux, where the GPU version is installed by default. On Linux, running `pip install torch --index-url https://download.pytorch.org/whl/cpu` first installs the smaller CPU version, which is sufficient for this model. `scikit-learn` is pinned to 1.8.0 because the model file stores scikit-learn 1.8.0 objects.

**6. Confirm the installation.**

```bash
hp-detect info
```

The training metadata stored in the model file is printed.

For later use, only step 3 and the activation line of step 4 (`.venv\Scripts\activate`) are repeated before running the model.

### Step 2. Prepare the meter data

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

- One year (12 consecutive months) per building. Longer data is cut to the most recent 12 months automatically.
- Readings at intervals of one hour or shorter (15, 30, or 60 minutes). The interval is detected per building.
- Total electricity consumption, not net of rooftop solar.
- Missing readings left blank or omitted, not recorded as 0.

**Data length.** The features summarize one year of data (for example, annual consumption, monthly means, and hours in each temperature range), and the model was trained on these one-year summaries.

- *Shorter than one year:* buildings are scored but flagged `low`. Accuracy decreases as months are removed, and most sharply when winter is missing, since heat pumps are distinguished by cold-weather load.
- *Longer than one year:* the package keeps the most recent 12 months of each building and notes `trimmed_to_last_12_months`. Without this step, annual totals and hour counts would be inflated and predictions would change (by 0.26 in probability on average when two identical years were supplied). `period_start` and `period_end` in the output show the period used.

Column names may differ from the example; they are specified in Step 4.

### Step 3. Prepare the outdoor temperature data

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

### Step 4. Write the configuration file

Column names and units differ between utilities, so the package needs to be told which column holds which information. The configuration file is a short text file (for example, `config.yaml`) that states this once, with one `setting: value` per line.

For example, if the two data files look like this:

```
meter_data.csv                        weather.csv
meter_id,read_time,usage              timestamp,temp_c
A1001,2023-01-01 00:15,0.42           2023-01-01 00:15,5.1
```

the configuration file is:

```yaml
id_column: meter_id            # column with the meter ID
timestamp_column: read_time    # column with the reading time
load_column: usage             # column with electricity use
weather_file: weather.csv      # temperature file: name only (same folder as this file) or full path, no double quotes
weather_temp_column: temp_c    # column with temperature in that file
```

The values on the right are replaced with the column names and file name of the actual data.

If temperature is a column of the meter data rather than a separate file, the last two lines are replaced by one line:

```yaml
temp_column: temp              # column with temperature in the meter data
```

**Matching the training data format.** The model was trained on electricity in kWh per reading and temperature in °C. Data in this format needs no further lines. Otherwise, the line below that matches the data is added, and the package converts it:

| If the data has | Add this line |
|---|---|
| Electricity in Wh | `load_unit: Wh` |
| Electricity in kW (average demand) | `load_unit: kW` |
| Temperature in °F | `temp_unit: F` |
| Times ending in a time zone offset, such as `2023-01-01T00:15:00-06:00` (not needed for times like `2023-01-01 00:15`) | `timezone: America/Chicago` (US Central); `America/New_York` (Eastern), `America/Denver` (Mountain), `America/Los_Angeles` (Pacific) |
| A time column in the weather file named something other than `timestamp` | `weather_timestamp_column: <column name>` |
| Temperatures from several weather stations in one weather file | `weather_key_column: station_id` (see below) |

With several weather stations, both files need a column naming the station, so that each meter is matched to the temperature of its own station:

```
weather.csv                               meter_data.csv
station_id,timestamp,temp_c               meter_id,station_id,read_time,usage
DAL,2023-01-01 00:15,5.1                  A1001,DAL,2023-01-01 00:15,0.42
FTW,2023-01-01 00:15,4.3                  A1002,FTW,2023-01-01 00:15,0.51
```

For example, for data in °F:

```yaml
id_column: meter_id
timestamp_column: read_time
load_column: usage
weather_file: weather.csv
weather_temp_column: temp_f
temp_unit: F
```

### Step 5. Run the prediction

To run the model, open a terminal (Command Prompt or PowerShell on Windows) in the folder that contains the data files and the configuration file, and run:

```bash
hp-detect predict meter_data.csv --config config.yaml --out predictions.csv
```

| Part | Meaning |
|---|---|
| `meter_data.csv` | Meter data file prepared in Step 2 |
| `--config config.yaml` | Configuration file written in Step 4 |
| `--out predictions.csv` | Name of the result file to be created |

File names or full paths of the actual files may be used. The command must be run in the same Python environment used for installation in Step 1; if `hp-detect` is not found, that environment is not active.

The package reads the files, converts them to the training format, computes the features, and applies the model. When it finishes, a summary is printed:

```
Decision threshold   : 0.35
Buildings scored     : <number of buildings>
Predicted heat pump  : <number> (<share>)
Low data quality     : <number>
```

Optional settings:

| Option | Use |
|---|---|
| `--workers 8` | Processes buildings in parallel; useful for large files |
| `--threshold 0.5` | Changes the decision threshold (default 0.35; see Step 6) |
| `--details` | Adds data checks and base-model probabilities to the output (see Step 6) |

The same prediction can be run from Python:

```python
from hp_detection import InputConfig, run

cfg = InputConfig(id_column="meter_id", timestamp_column="read_time",
                  load_column="usage", weather_file="weather.csv",
                  weather_temp_column="temp_c")
result = run("meter_data.csv", cfg)
```

### Step 6. Review the output

`predictions.csv` contains one row per building:

| Column | Contents |
|---|---|
| `building_id` | Meter ID from the input |
| `hp_probability` | Probability of a heat pump (0-1) |
| `hp_predicted` | 1 = likely heat pump, 0 = likely not. A building is marked 1 when `hp_probability` is 0.35 or higher |
| `quality_flag` | `ok` = the data covers a full year with few gaps. `low` = the data is incomplete, so the result is less reliable |
| `quality_notes` | Why the data is incomplete (see below) |

| `quality_notes` value | Meaning |
|---|---|
| `less_than_one_year` | Less than one year of data |
| `missing_months` | One or more months have no data |
| `load_gaps` | Electricity readings are missing for more than 10% of hours |
| `temperature_gaps` | Temperature is missing for more than 10% of hours |
| `negative_load` | Some electricity readings are negative (often rooftop solar) |
| `duplicate_timestamps` | The same time appears twice, usually at the daylight saving change. For information only; the flag stays `ok` |
| `trimmed_to_last_12_months` | More than one year was supplied, and only the last 12 months were used. For information only; the flag stays `ok` |

Predictions cannot be scored without known heat pump status. The following checks are recommended:

- Buildings flagged `low` should be treated with caution.
- The overall predicted heat pump share can be compared with published regional statistics.
- `hp_probability` is suited to ranking buildings; `hp_predicted` applies a fixed threshold.

**Additional columns.** With `--details`, the file also contains the data checks below, which help trace unexpected results:

| Column | Contents |
|---|---|
| `p_gradient_boosting`, `p_mlp`, `p_cnn` | Probabilities of the three base models; `hp_probability` is their average |
| `annual_kwh` | Total electricity over the data period. Values several times higher or lower than expected typically indicate a unit setting error |
| `interval_minutes` | Detected reading interval |
| `period_start`, `period_end` | First and last hour of the data used |
| `span_days`, `load_coverage_pct`, `temp_coverage_pct` | Data period and share of hours with data |

#### Decision threshold (0.35)

`hp_predicted` is 1 when `hp_probability` is at least the decision threshold. The package uses 0.35 rather than the conventional 0.5.

**Why 0.35.** The model was trained in Dallas County. On all 6,063 ResStock buildings in the six adjacent counties, it ranks buildings as well as in Dallas County (ROC-AUC 0.969), but its probabilities are lower, so at 0.5 many heat pumps fall below the threshold. At 0.35, both areas perform alike:

| Threshold | Area | Precision | Recall | F1-score |
|---:|---|---:|---:|---:|
| 0.5 | Dallas County (cross-validation) | 0.859 | 0.828 | 0.843 |
| 0.5 | Six adjacent counties | 0.899 | **0.585** | 0.709 |
| 0.35 | Dallas County (cross-validation) | 0.810 | 0.891 | 0.848 |
| 0.35 | Six adjacent counties | 0.852 | 0.838 | 0.845 |

Precision is the share of predicted heat pumps that are correct; recall is the share of actual heat pumps that are found.

**Changing the threshold.** `--threshold` sets a different value. A lower threshold finds more heat pumps but includes more buildings without one; a higher threshold does the opposite. For example, `--threshold 0.5` suits a program where each false positive is costly. `hp_probability` itself does not depend on the threshold.

Results for each county are in `2. Model_Development/3. Output/3. External Validation/`.

`--save-inputs DIR` also writes the computed features and load profiles.

---

## Notes

Performance on other datasets may differ because of differences in climate, building stock, heat-pump prevalence, data resolution, missing data, and other dataset characteristics. The model was trained on simulated buildings in a single county. External validation covers simulated buildings in the six adjacent counties only; validation against utility customers with known heating equipment is recommended before operational use.

Multi-family buildings with five or more units are the weakest building type (ROC-AUC about 0.90, compared with 0.99 for single-family detached homes).

Other electric heating systems (electric furnace, baseboard, boiler) are the most common source of false positives. Error rates by heating type are listed in `2. Model_Development/3. Output/2. Results/misclassification_by_type.csv`.

Only load serialized model files such as `joblib` from trusted sources.
