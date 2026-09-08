# HP_detection

Heat pump detection from building-level electricity consumption and
outdoor air temperature.

Given a year of interval meter data and a co-located temperature series,
the pipeline classifies whether each household has a heat pump. 

---

## Repository layout

| Path | Contents |
|---|---|
| `Code/` | The four pipeline scripts and their documentation |
| `Data_Source/Dallas_County_Residential/` | Sample input files showing the expected schema |

Start with [`Code/Code_README.md`](Code/Code_README.md) for how to run the
pipeline, what the inputs must look like, and what each output file
contains.

---

## Pipeline

Four scripts, run in order:

1. `01_build_features.py` — 102 statistical features per building
2. `02_build_profiles.py` — four 24-hour load profiles per building
3. `03_train_evaluate.py` — stratified 5-fold cross-validation of a
   three-model ensemble, plus all result tables
4. `04_permutation_importance.py` — ROC-AUC drop per input

---

## Data

Developed and evaluated on ResStock 2025 Release 1, Dallas County, TX.

The full building set is not included in this repository. Ten sample
timeseries files and a metadata CSV are provided under
`Data_Source/Dallas_County_Residential/` so the expected file naming, column names, and
interval structure can be inspected. See
[`Data_Source/Data_README.md`](Data_Source/Data_README.md) for the
schema and for what to change when adapting to a different one.

---

## Requirements

Python with numpy, pandas, pyarrow, scikit-learn, and torch. Pinned
versions in `Code/requirements.txt`. Runs on CPU or GPU.
