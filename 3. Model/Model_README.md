# Final Model Bundle

This folder contains the final trained heat-pump detection model for reuse on
new datasets after running the feature and profile extraction steps.

## Model file

`hp_detection_model.joblib`

The bundle is produced by `05_train_final_model.py` after fitting the three
base models on the complete Dallas County development dataset.

The single bundle contains:

- fitted Gradient Boosting model,
- fitted MLP model,
- fitted StandardScaler,
- HybridCNN learned parameters,
- statistical feature order,
- load-profile order,
- decision threshold, and
- basic training and software metadata.

The three model probabilities are averaged to produce the final soft-voting
heat-pump probability.

## Training

First run scripts 01 and 02 to create:

- `dallas_features_groupA_E.csv`
- `dallas_cnn_profiles.csv`

Then train the final model bundle:

```bat
python "2. Code\05_train_final_model.py" ^
  --work-dir "outputs" ^
  --metadata "path\to\TX_upgrade0_Residential.csv" ^
  --model-out "3. Model\hp_detection_model.joblib"
```

The final bundle is fitted on all available development buildings. It is
intended for application to new data, not for reporting evaluation
performance.

Use the 5-fold cross-validation results from `03_train_evaluate.py` for
performance reporting.

## Prediction on new data

For a new dataset, first run scripts 01 and 02 to generate the same 102
statistical features and four 24-hour profiles. Then run:

```bat
python "2. Code\06_predict.py" ^
  --model "3. Model\hp_detection_model.joblib" ^
  --features "path\to\new_features.csv" ^
  --profiles "path\to\new_profiles.csv" ^
  --output "utility_predictions.csv"
```

The output contains:

- `bldg_id`
- `gb_prob`
- `mlp_prob`
- `cnn_prob`
- `ensemble_prob`
- `predicted_label`

`predicted_label = 1` indicates a heat-pump prediction.

## Important

The model was trained on the Dallas County ResStock development dataset.
Performance on utility data may differ because of climate, building stock,
heat-pump prevalence, data resolution, missing data, and other dataset
characteristics.

Only load model bundles from trusted sources. Python model serialization
formats such as joblib should not be loaded from untrusted files.
