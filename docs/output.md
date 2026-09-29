# Output

`hp-detect predict` writes one CSV file with one row per home.
Example: [`examples/predictions.csv`](../examples/predictions.csv).

## Main columns

| Column | Meaning |
|---|---|
| `building_id` | Meter ID from your data |
| `hp_probability` | Chance the home has a heat pump (0 to 1) |
| `hp_predicted` | 1 if `hp_probability` is 0.5 or higher |
| `quality_flag` | `ok`, or `low` if the data has problems |
| `quality_notes` | Why the flag is `low` (see below) |

## Data checks

| Column | Meaning |
|---|---|
| `annual_kwh` | Total electricity. Very high or low values suggest a unit error |
| `interval_minutes` | Detected reading interval |
| `span_days` | Days of data |
| `load_coverage_pct` | Share of hours with electricity data |
| `temp_coverage_pct` | Share of hours with temperature data |

`quality_notes` values:

| Note | Meaning |
|---|---|
| `less_than_one_year` | Under 350 days of data |
| `missing_months` | Some months have no data |
| `load_gaps` | Electricity data missing for over 10% of hours |
| `temperature_gaps` | Temperature missing for over 10% of hours |
| `negative_load` | Negative readings (often rooftop solar) |
| `duplicate_timestamps` | Repeated times, usually daylight saving. For information only |

## Other columns

`p_gradient_boosting`, `p_mlp`, `p_cnn`: the three parts of the model.
`hp_probability` is their average.

Add `--save-inputs DIR` to also save the values the model computed from
your data.

## Checking results

You cannot score the results without knowing which homes have heat
pumps, but you can:

- Be cautious with homes flagged `low`.
- Compare the share of predicted heat pumps with published statistics
  for your area. A big difference may mean a unit or setting error.
- Use `hp_probability` to rank homes, not only the 0/1 prediction.
