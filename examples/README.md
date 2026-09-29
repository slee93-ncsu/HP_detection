# Example

Try the model on four homes:

```bash
hp-detect predict examples/meter_data.csv --config examples/config.yaml --out predictions.csv
```

| File | What it is |
|---|---|
| `meter_data.csv` | Electricity every 15 minutes, 4 homes, 1 year |
| `weather.csv` | Outdoor temperature (°C) |
| `config.yaml` | Tells the model which column is which |
| `predictions.csv` | The result |
| `config_template.yaml` | All settings, for your own data |

## About these homes

Simulated homes from Tarrant County, Texas, **not** used to train the
model. Their real heating systems:

| Home | Real heating | Probability | Predicted |
|---|---|---:|---|
| M352201 | Heat pump | 0.61 | Heat pump |
| M380181 | Heat pump | 0.61 | Heat pump |
| M498044 | Gas furnace | 0.00 | Not heat pump |
| M498133 | Electric furnace | 0.39 | Not heat pump |

Four homes only show what the output looks like. For accuracy, see the
[model card](../docs/model_card.md).
