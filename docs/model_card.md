# Model Card

## What it does

Estimates the chance that a home's main heating system is a heat pump,
from one year of electricity use and outdoor temperature.

**Good for:** finding likely heat pump homes across many meters, for
program planning or outreach.
**Not for:** confirming the equipment in a single home.

## How it works

The model turns each home's data into 102 summary numbers (for example,
how much electricity use rises in cold weather) and four daily usage
curves. Three machine learning models each give a probability, and the
final answer is their average.

## Training data

- 4,005 homes in Dallas County, Texas
- NREL ResStock 2025 (**simulated** homes, not real meters), weather year 2018
- A home counts as a heat pump if its main heating is a ducted heat pump
  (`Electricity ASHP`) or a ductless mini-split (`Electricity MSHP`).
  1,071 homes (27%) had one.

## Accuracy

Tested with 5-fold cross-validation (each home is predicted by a model
that did not see it).

| Measure | Result |
|---|---|
| Accuracy | 92% of homes classified correctly |
| Recall | Finds 83% of heat pumps |
| Precision | 86% of homes it flags really have one |
| ROC-AUC | 0.97 (1.0 = perfect ranking) |

Out of 4,005 homes:

| | Predicted no HP | Predicted HP |
|---|---:|---:|
| **No heat pump** | 2,788 | 146 |
| **Heat pump** | 184 | 887 |

## Common mistakes

- **Other electric heating is sometimes mistaken for a heat pump**
  (electric furnace 8%, baseboard 10%). Their use also rises in cold weather.
- **Gas-heated homes are almost never mistaken** (0.3%).
- **About 1 in 6 ducted heat pumps is missed.**

## Limits

- Trained on **one county in Texas**. Other climates are not tested.
- Trained on **simulated** data. Real meters are messier.
- Needs **a full year** of data.
- Electricity must be **total use**, not net of solar.
- If heat pumps are much more or less common in your area than 27%, the
  number of predicted heat pumps may be off. Rank by `hp_probability`.

Full results: [`model_development/`](../model_development/README.md).

## Security

The model file (`.joblib`) can run code when opened. Only use model
files from trusted sources.
