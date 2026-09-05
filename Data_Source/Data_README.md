# Sample data

Reference files showing the input format the pipeline expects. The full
building set is too large to include; a small sample of timeseries files
is provided so the file naming, column names, and interval structure can
be checked before running on your own data.

Running the pipeline on this sample will produce output files, but the
metrics will not be meaningful at this sample size.

---

## Timeseries files

Ten parquet files, one per building, named `{bldg_id}-{upgrade}.parquet`.

| Property | Value |
|---|---|
| Rows per file | 35,040 |
| Interval | 15 minutes |
| Period | One calendar year |
| Columns | 192 |

The pipeline reads three of the 192 columns:

| Column | Type | Meaning |
|---|---|---|
| `timestamp` | datetime | Interval end |
| `out.electricity.total.energy_consumption..kwh` | float | Total electricity, kWh per interval |
| `out.outdoor_air_drybulb_temp..c` | float | Outdoor drybulb temperature, °C |

The files also contain per-end-use columns, including heat pump and
backup heating circuits. These are not used: they are a direct function
of the label.

Outdoor temperature is embedded in each building file, so no separate
weather file is needed. If your data keeps weather separately, join it
to the load timeseries before running script 01, or adjust
`load_building()` in scripts 01 and 02.

---

## Metadata

One CSV with one row per building. The pipeline reads two columns:

| Column | Meaning |
|---|---|
| `bldg_id` | Building identifier, matching the parquet filenames |
| `in.hvac_heating_type_and_fuel` | Heating equipment and fuel |

The label is derived from the heating type column. A building is
positive if its value appears in `HP_HEATING_TYPES`, defined at the top
of `03_train_evaluate.py`. Edit that set to match the equipment labels
in your own metadata.

The metadata may cover more buildings than the timeseries directory; the
two are inner-joined on `bldg_id`.

---

## Adapting to a different schema

| To change | Edit |
|---|---|
| Column names | `COL_TIMESTAMP`, `COL_TOTAL_LOAD`, `COL_OUTDOOR_TEMP` in scripts 01 and 02 |
| Filename pattern | `FILENAME_PATTERN` in scripts 01 and 02 |
| Label definition | `HP_HEATING_TYPES` and the heating type column name in script 03 |
| Temperature unit | `to_fahrenheit()` in script 01 |

Any interval at or below one hour works; the scripts resample to hourly,
summing load and averaging temperature. Seasonal and monthly features
assume a full year of data.
