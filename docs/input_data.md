# Input Data

You provide two files: **meter data** and **weather data**. Both can be
CSV or parquet. Column names can be anything; you list them in a config
file.

## 1. Meter data

All homes in one file, one row per reading.

| Column | Example | Unit |
|---|---|---|
| Meter ID | `A1001` | - |
| Time | `2023-01-01 00:15` | local time |
| Electricity | `0.42` | kWh per reading |

```
meter_id,read_time,usage
A1001,2023-01-01 00:15,0.42
A1001,2023-01-01 00:30,0.38
```

## 2. Weather data

| Column | Example | Unit |
|---|---|---|
| Time | `2023-01-01 00:15` | local time |
| Outdoor temperature | `5.1` | °C |

```
timestamp,temp_c
2023-01-01 00:15,5.1
2023-01-01 00:30,4.7
```

If temperature is already in the meter file, you can skip this file
(set `temp_column` instead of `weather_file`).

## Rules

- **One full year** per home.
- Readings **every hour or more often** (15, 30, or 60 minutes).
- **Total** electricity use, not net of rooftop solar.
- Missing readings: **leave them out or blank. Do not use 0.**
- Outdoor air temperature, not indoor.

## 3. Config file

Tells the model which column is which. Example
([`examples/config.yaml`](../examples/config.yaml)):

```yaml
format: long
id_column: meter_id
timestamp_column: read_time
load_column: usage
weather_file: weather.csv
weather_temp_column: temp_c
```

Relative paths are read from the config file's folder.

### If your units are different

The defaults match the training data: **kWh**, **°C**, and timestamps at
the **end** of each reading period. Add these lines only if yours differ:

| Your data | Add |
|---|---|
| Wh instead of kWh | `load_unit: Wh` |
| kW (average demand) instead of kWh | `load_unit: kW` |
| °F instead of °C | `temp_unit: F` |
| Time marks the **start** of each period (e.g. 00:00 for 00:00-00:15) | `timestamp_convention: start` |
| Times include a UTC offset (e.g. `-06:00`) | `timezone: America/Chicago` |
| Several weather stations | `weather_key_column: station_id` (column in both files) |

All settings, with comments: [`examples/config_template.yaml`](../examples/config_template.yaml).

## Details

- **Reading interval** is detected automatically for each home. Hourly or
  shorter works; daily or monthly data does not.
- All data is combined into **hourly** values before prediction.
- **Daylight saving time:** the model uses standard time all year. If
  your times include a UTC offset, set `timezone` and this is handled
  for you.
- If a home's reading interval changes during the year (e.g. hourly,
  then 15-min), supply its data in kWh with end-of-period timestamps.
- **Quick checks:** a typical home uses 5,000-20,000 kWh per year
  (`annual_kwh` in the output). January temperatures in most of the U.S.
  are -5 to 10 °C (20-50 °F).
