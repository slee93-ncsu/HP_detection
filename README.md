# HP_identification

Finds which homes likely have a **heat pump**, using only smart meter
electricity data and outdoor temperature. You do not need to know which
homes have heat pumps.

## What you need

- One year of electricity readings per home (every 15, 30, or 60 minutes)
- Outdoor temperature for the same year
- Python 3.11 or later

## How to run

```bash
pip install .
hp-detect predict examples/meter_data.csv --config examples/config.yaml --out predictions.csv
```

This runs the included example. For your own data:

1. Save your meter data and temperature as CSV files ([format](docs/input_data.md)).
2. Copy [`examples/config.yaml`](examples/config.yaml) and change the column names.
3. Run `hp-detect predict your_data.csv --config your_config.yaml --out predictions.csv`.

## What you get

A `predictions.csv` file with one row per home:

| Column | Meaning |
|---|---|
| `hp_probability` | Chance the home has a heat pump (0 to 1) |
| `hp_predicted` | 1 = likely heat pump, 0 = likely not |
| `quality_flag` | `ok`, or `low` if the data has problems |

More: [output guide](docs/output.md).

## How accurate is it?

Tested on 4,005 simulated homes in Dallas County, Texas: **92% accuracy**,
and it finds **83% of heat pumps**. Results may differ in other regions.
See the [model card](docs/model_card.md).

## Files

| Folder | What's inside |
|---|---|
| `hp_detection/` | The model and code |
| `examples/` | Sample input and output |
| `docs/` | Input guide, output guide, model card |
| `tests/` | Checks that the install works (`pip install ".[test]"`, then `pytest`) |
| `model_development/` | How the model was built (for reference) |
