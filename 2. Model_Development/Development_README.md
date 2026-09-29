# Model Development

This folder documents how the heat pump detection model in `1. Model/` was developed and evaluated. It is provided for reference and reproducibility; it is not required to apply the model to new data.

## Folder Layout

| Path | Contents | Documentation |
|---|---|---|
| `1. Data_Source/` | Sample building timeseries and the heating-system metadata used to define labels | `Data_README.md` |
| `2. Code/` | Feature extraction, model evaluation, feature-importance, and final-model training scripts | `Code_README.md` |
| `3. Output/` | Processed model inputs and evaluation results | `Output_README.md` |

## Workflow

1. `1. Data_Source/` provides building-level electricity consumption, outdoor temperature, and heating-system metadata from ResStock 2025 Release 1 for Dallas County, Texas.
2. `2. Code/01_build_features.py` and `02_build_profiles.py` convert each building's timeseries into 102 statistical features and four 24-hour load profiles.
3. `2. Code/03_train_evaluate.py` evaluates Gradient Boosting, MLP, HybridCNN, and their soft-voting ensemble with stratified 5-fold cross-validation. `04_permutation_importance.py` measures input importance.
4. `2. Code/05_train_final_model.py` fits the ensemble on the complete dataset and saves the model bundle used in `1. Model/`.
5. `3. Output/` stores the model inputs and evaluation results produced by steps 2 and 3.

Performance should be reported from the cross-validation results in `3. Output/`, not from the final model fitted on the complete dataset.
