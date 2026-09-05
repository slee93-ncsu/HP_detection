"""
Train and evaluate a three-model ensemble with stratified 5-fold CV.

Inputs:
    dallas_features_groupA_E.csv   from script 01 (bldg_id + 102 features)
    dallas_cnn_profiles.csv        from script 02 (bldg_id + 96 columns)
    metadata CSV                   bldg_id + in.hvac_heating_type_and_fuel

The three inputs are inner-joined on bldg_id, so the metadata may cover a
wider area than the timeseries.

Models, all trained per fold:
    GradientBoosting  raw 102 features
    MLP               standardized 102 features
    HybridCNN         (4, 24) profile tensor + standardized 102 features
    SoftVoting        mean of the three probabilities
    Stacking          optional, --with-stacking

Outputs, written to --work-dir:
    predictions.csv, metrics_per_fold.csv, metrics_summary.csv,
    model_comparison.csv, confusion_matrix.csv, model_ablation.csv,
    model_correlation.csv, error_overlap.csv,
    misclassification_by_type.csv, run_summary.json

The StandardScaler is fitted inside the fold loop on training rows only.
Cross-validation runs once and every table is derived from that one run.

Usage:
    python 03_train_evaluate.py --metadata "path/to/TX_upgrade0.csv"
    python 03_train_evaluate.py --metadata "..." --with-stacking
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

WORK_DIR = Path(os.environ.get("HP_WORK_DIR", "outputs"))

METADATA_CSV = Path(os.environ.get(
    "HP_METADATA_CSV",
    str(Path(os.environ.get("HP_DATA_ROOT", "data")) / "TX_upgrade0.csv"),
))

# Positive class. Everything else, including electric resistance heating,
# is negative.
HP_HEATING_TYPES = {"Electricity ASHP", "Electricity MSHP"}

PROFILE_NAMES = ("winter_weekday", "summer_weekday", "shoulder_weekday", "peak_day")
PROFILE_LENGTH = 24
N_CHANNELS = len(PROFILE_NAMES)

RANDOM_STATE = 42
N_FOLDS = 5
DECISION_THRESHOLD = 0.5

GB_PARAMS = {"random_state": RANDOM_STATE}
MLP_PARAMS = {"hidden_layer_sizes": (64, 32), "max_iter": 300,
              "random_state": RANDOM_STATE}

CNN_EPOCHS = 60
CNN_LR = 1e-3
CNN_BATCH_SIZE = 64
CNN_DROPOUT = 0.2

STACKING_INNER_FOLDS = 5
STACKING_META_PARAMS = {"class_weight": "balanced", "max_iter": 1000}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seeds(seed: int = RANDOM_STATE) -> None:
    """Seed numpy and torch. Results are reproducible on the same machine
    and library versions; PyTorch does not guarantee bit-identical output
    across different hardware."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class HybridCNN(nn.Module):
    """Two-branch classifier.

    Conv branch: Conv1d(4->16) -> pool -> Conv1d(16->32) -> pool, applied
    over the 24-hour axis with the four profiles as channels. Output is
    32 x 6 flattened to 192.

    Dense branch: n_stat_features -> 64 -> 32.

    The two are concatenated (224) and passed through 64 -> dropout -> 1.

    forward(x_profiles, x_stat) takes tensors of shape (batch, 4, 24) and
    (batch, n_stat_features) and returns logits of shape (batch,).
    """

    def __init__(self, n_stat_features: int):
        super().__init__()

        self.conv1 = nn.Conv1d(N_CHANNELS, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        conv_out = 32 * (PROFILE_LENGTH // 4)  # 24 -> 12 -> 6

        self.stat_fc = nn.Sequential(
            nn.Linear(n_stat_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(conv_out + 32, 64), nn.ReLU(),
            nn.Dropout(CNN_DROPOUT), nn.Linear(64, 1),
        )

    def forward(self, x_profiles, x_stat):
        c = self.pool(self.relu(self.conv1(x_profiles)))
        c = self.pool(self.relu(self.conv2(c)))

        combined = torch.cat([c.flatten(start_dim=1), self.stat_fc(x_stat)], dim=1)

        return self.head(combined).squeeze(1)


class ProfileStatDataset(Dataset):
    """Yields (profiles, stat) or (profiles, stat, label) per building."""

    def __init__(self, x_profiles, x_stat, y=None):
        self.x_profiles = torch.tensor(x_profiles, dtype=torch.float32)
        self.x_stat = torch.tensor(x_stat, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.x_stat)

    def __getitem__(self, i):
        if self.y is None:
            return self.x_profiles[i], self.x_stat[i]
        return self.x_profiles[i], self.x_stat[i], self.y[i]


def train_cnn(x_profiles, x_stat_scaled, y, device, verbose=True) -> HybridCNN:
    """Train one HybridCNN for CNN_EPOCHS.

    x_profiles  (n, 4, 24) float32
    x_stat_scaled  (n, n_features) float32, already scaled by the caller
    y  (n,) float32 in {0, 1}
    """
    model = HybridCNN(x_stat_scaled.shape[1]).to(device)
    dataset = ProfileStatDataset(x_profiles, x_stat_scaled, y)
    loader = DataLoader(dataset, batch_size=CNN_BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=CNN_LR)
    criterion = nn.BCEWithLogitsLoss()

    model.train()

    for epoch in range(CNN_EPOCHS):
        epoch_loss = 0.0

        for profiles, stat, target in loader:
            optimizer.zero_grad()
            loss = criterion(model(profiles.to(device), stat.to(device)),
                             target.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(target)

        if verbose and (epoch == 0 or (epoch + 1) % 20 == 0):
            print(f"    epoch {epoch + 1:>3}/{CNN_EPOCHS}: "
                  f"loss={epoch_loss / len(dataset):.5f}")

    model.eval()

    return model


def predict_cnn(model, x_profiles, x_stat_scaled, device) -> np.ndarray:
    """Return positive-class probabilities, shape (n,)."""
    model.eval()
    loader = DataLoader(ProfileStatDataset(x_profiles, x_stat_scaled),
                        batch_size=256, shuffle=False)
    out = []

    with torch.no_grad():
        for profiles, stat in loader:
            logits = model(profiles.to(device), stat.to(device))
            out.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(out)


def soft_vote(*probabilities) -> np.ndarray:
    """Element-wise mean of any number of equal-length probability arrays."""
    return np.mean(np.vstack(probabilities), axis=0)


def train_stacking_meta(x_stat, x_profiles, y, device, verbose=True):
    """Fit a logistic meta-learner on out-of-fold base-model probabilities.

    Runs an inner StratifiedKFold over the outer training split so the
    meta-learner never sees probabilities the base models produced on
    their own training rows. Trains 3 * STACKING_INNER_FOLDS models,
    which is the main cost of --with-stacking.

    Returns a fitted LogisticRegression expecting a (n, 3) array of
    [gb_prob, mlp_prob, cnn_prob].
    """
    splitter = StratifiedKFold(n_splits=STACKING_INNER_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE)
    oof = np.zeros((len(x_stat), 3))

    for fold, (train_idx, val_idx) in enumerate(splitter.split(x_stat, y), start=1):
        if verbose:
            print(f"    inner fold {fold}/{STACKING_INNER_FOLDS}")

        scaler = StandardScaler().fit(x_stat[train_idx])
        train_scaled = scaler.transform(x_stat[train_idx])
        val_scaled = scaler.transform(x_stat[val_idx])

        gb = GradientBoostingClassifier(**GB_PARAMS).fit(
            x_stat[train_idx], y[train_idx])
        oof[val_idx, 0] = gb.predict_proba(x_stat[val_idx])[:, 1]

        mlp = MLPClassifier(**MLP_PARAMS).fit(train_scaled, y[train_idx])
        oof[val_idx, 1] = mlp.predict_proba(val_scaled)[:, 1]

        cnn = train_cnn(x_profiles[train_idx], train_scaled,
                        y[train_idx].astype(np.float32), device, verbose=False)
        oof[val_idx, 2] = predict_cnn(cnn, x_profiles[val_idx], val_scaled, device)

    return LogisticRegression(**STACKING_META_PARAMS).fit(oof, y)


def normalize_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing bldg_id, cast it to int, and keep the first
    row per bldg_id. Duplicates would multiply rows through the joins."""
    frame = frame.dropna(subset=["bldg_id"]).copy()
    frame["bldg_id"] = frame["bldg_id"].astype(int)

    return frame.drop_duplicates(subset="bldg_id", keep="first")


def profile_columns() -> list:
    """The 96 profile column names, matching script 02's output order."""
    return [f"{n}_h{h:02d}" for n in PROFILE_NAMES for h in range(PROFILE_LENGTH)]


def build_dataset(features_csv, profiles_csv, metadata_csv, verbose=True) -> dict:
    """Join the three inputs and return model-ready arrays.

    Returns a dict with:
        x_stat         (n, 102) float32
        x_profiles     (n, 4, 24) float32
        y              (n,) int64, 1 = heat pump
        bldg_ids       (n,) int
        heating_types  (n,) str
        stat_columns   list of 102 column names, matching x_stat order

    Rows are sorted by bldg_id so fold assignment does not depend on file
    row order. NaN and inf are replaced with 0 after merging.
    """
    features = normalize_ids(pd.read_csv(features_csv, low_memory=False))
    profiles = normalize_ids(pd.read_csv(profiles_csv, low_memory=False))

    meta = pd.read_csv(metadata_csv, low_memory=False)

    if "bldg_id" not in meta.columns:
        meta = meta.rename(columns={meta.columns[0]: "bldg_id"})

    meta = normalize_ids(meta)
    meta["true_label"] = meta["in.hvac_heating_type_and_fuel"].isin(
        HP_HEATING_TYPES).astype(int)
    meta = meta[["bldg_id", "in.hvac_heating_type_and_fuel", "true_label"]]

    common = set(features["bldg_id"]) & set(profiles["bldg_id"]) & set(meta["bldg_id"])

    if verbose:
        print(f"Buildings with features : {len(features):,}")
        print(f"Buildings with profiles : {len(profiles):,}")
        print(f"Buildings in metadata   : {len(meta):,}")
        print(f"Common to all three     : {len(common):,}")

    if not common:
        raise SystemExit("No buildings present in all three inputs.")

    merged = (features[features["bldg_id"].isin(common)]
              .merge(profiles, on="bldg_id")
              .merge(meta, on="bldg_id")
              .sort_values("bldg_id")
              .reset_index(drop=True))

    stat_cols = [c for c in features.columns if c != "bldg_id"]
    prof_cols = profile_columns()

    missing = [c for c in prof_cols if c not in merged.columns]
    if missing:
        raise SystemExit(f"Profile CSV missing columns, e.g. {missing[:5]}")

    for cols in (stat_cols, prof_cols):
        merged[cols] = merged[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    data = {
        "x_stat": merged[stat_cols].to_numpy(dtype=np.float32),
        "x_profiles": merged[prof_cols].to_numpy(dtype=np.float32).reshape(
            -1, N_CHANNELS, PROFILE_LENGTH),
        "y": merged["true_label"].to_numpy(dtype=np.int64),
        "bldg_ids": merged["bldg_id"].to_numpy(dtype=int),
        "heating_types": merged["in.hvac_heating_type_and_fuel"].astype(str).to_numpy(),
        "stat_columns": stat_cols,
    }

    if verbose:
        print(f"\nInput matrix   : {data['x_stat'].shape}")
        print(f"Profile tensor : {data['x_profiles'].shape}")
        print(f"Positive rate  : {data['y'].mean() * 100:.2f}%")

    return data


def compute_metrics(y_true, y_prob, threshold=DECISION_THRESHOLD) -> dict:
    """Return accuracy, precision, recall, f1, roc_auc, pr_auc.

    Thresholded metrics use `threshold`; roc_auc and pr_auc use the raw
    probabilities.
    """
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def run_cross_validation(data, with_stacking=False, verbose=True):
    """Run stratified 5-fold CV and return (predictions, fold_artifacts).

    predictions: one row per building, with each model's probability, the
        ensemble label, and the fold it was held out in.
    fold_artifacts: list of dicts with the fitted gb_model, mlp_model,
        cnn_model, scaler, test_index, and ensemble_prob for each fold.
        Script 04 reuses these instead of retraining.
    """
    set_seeds()
    device = get_device()

    if verbose:
        print(f"Device: {device}")

    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                               random_state=RANDOM_STATE)
    artifacts, frames = [], []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(data["x_stat"], data["y"]), start=1
    ):
        if verbose:
            print(f"\n{'=' * 60}\nFold {fold}/{N_FOLDS}\n{'=' * 60}")

        x_stat_train, x_stat_test = data["x_stat"][train_idx], data["x_stat"][test_idx]
        x_prof_train = data["x_profiles"][train_idx]
        x_prof_test = data["x_profiles"][test_idx]
        y_train, y_test = data["y"][train_idx], data["y"][test_idx]

        # Fitted on training rows only.
        scaler = StandardScaler().fit(x_stat_train)
        train_scaled = scaler.transform(x_stat_train)
        test_scaled = scaler.transform(x_stat_test)

        if verbose:
            print("  Gradient boosting...")
        gb = GradientBoostingClassifier(**GB_PARAMS).fit(x_stat_train, y_train)
        gb_prob = gb.predict_proba(x_stat_test)[:, 1]

        if verbose:
            print("  MLP...")
        mlp = MLPClassifier(**MLP_PARAMS).fit(train_scaled, y_train)
        mlp_prob = mlp.predict_proba(test_scaled)[:, 1]

        if verbose:
            print("  Hybrid CNN...")
        cnn = train_cnn(x_prof_train, train_scaled, y_train.astype(np.float32),
                        device, verbose=verbose)
        cnn_prob = predict_cnn(cnn, x_prof_test, test_scaled, device)

        ensemble_prob = soft_vote(gb_prob, mlp_prob, cnn_prob)

        stacking_prob = None

        if with_stacking:
            if verbose:
                print("  Stacking meta-learner...")
            meta = train_stacking_meta(x_stat_train, x_prof_train, y_train,
                                       device, verbose=verbose)
            stacking_prob = meta.predict_proba(
                np.column_stack([gb_prob, mlp_prob, cnn_prob]))[:, 1]

        if verbose:
            m = compute_metrics(y_test, ensemble_prob)
            print(f"  Ensemble ROC-AUC = {m['roc_auc']:.4f}  "
                  f"Accuracy = {m['accuracy']:.4f}")

        artifacts.append({
            "fold": fold, "test_index": test_idx, "ensemble_prob": ensemble_prob,
            "gb_model": gb, "mlp_model": mlp, "cnn_model": cnn, "scaler": scaler,
        })

        frame = pd.DataFrame({
            "bldg_id": data["bldg_ids"][test_idx],
            "heating_type": data["heating_types"][test_idx],
            "true_label": y_test,
            "gb_prob": gb_prob, "mlp_prob": mlp_prob, "cnn_prob": cnn_prob,
            "ensemble_prob": ensemble_prob,
            "pred_label": (ensemble_prob >= DECISION_THRESHOLD).astype(int),
            "fold": fold,
        })

        if stacking_prob is not None:
            frame["stacking_prob"] = stacking_prob

        frames.append(frame)

    predictions = (pd.concat(frames, ignore_index=True)
                   .sort_values("bldg_id").reset_index(drop=True))
    predictions["correct"] = predictions["true_label"] == predictions["pred_label"]

    return predictions, artifacts


def summarize_folds(predictions: pd.DataFrame):
    """Return (per_fold, summary): ensemble metrics for each fold, and their
    mean and standard deviation."""
    rows = []

    for fold, group in predictions.groupby("fold"):
        m = compute_metrics(group["true_label"].to_numpy(),
                            group["ensemble_prob"].to_numpy())
        m["fold"] = fold
        rows.append(m)

    per_fold = pd.DataFrame(rows).set_index("fold").sort_index()

    return per_fold, pd.DataFrame({"mean": per_fold.mean(), "std": per_fold.std()})


def model_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    """Metrics for each model, indexed by model name. Stacking is included
    only if the run used --with-stacking."""
    y_true = predictions["true_label"].to_numpy()

    models = {
        "GradientBoosting": "gb_prob",
        "MLP": "mlp_prob",
        "HybridCNN": "cnn_prob",
        "SoftVoting": "ensemble_prob",
    }

    if "stacking_prob" in predictions.columns:
        models["Stacking"] = "stacking_prob"

    rows = []

    for name, column in models.items():
        m = compute_metrics(y_true, predictions[column].to_numpy())
        m["model"] = name
        rows.append(m)

    return pd.DataFrame(rows).set_index("model")


def pooled_confusion(predictions: pd.DataFrame) -> pd.DataFrame:
    """2x2 confusion matrix over all out-of-fold predictions at
    DECISION_THRESHOLD, with labelled rows and columns."""
    matrix = confusion_matrix(predictions["true_label"], predictions["pred_label"])

    return pd.DataFrame(matrix,
                        index=["Actual: Non-HP", "Actual: HP"],
                        columns=["Predicted: Non-HP", "Predicted: HP"])


def model_ablation(predictions: pd.DataFrame) -> pd.DataFrame:
    """Metrics for all seven non-empty subsets of {GB, MLP, CNN}, each
    combined by soft voting. Sorted by roc_auc descending."""
    y_true = predictions["true_label"].to_numpy()
    columns = {"GB": "gb_prob", "MLP": "mlp_prob", "CNN": "cnn_prob"}

    rows = []

    for combo in [("GB",), ("MLP",), ("CNN",), ("GB", "MLP"), ("GB", "CNN"),
                  ("MLP", "CNN"), ("GB", "MLP", "CNN")]:
        m = compute_metrics(y_true, soft_vote(
            *[predictions[columns[c]].to_numpy() for c in combo]))
        m["combo"] = " + ".join(combo)
        m["n_models"] = len(combo)
        rows.append(m)

    return pd.DataFrame(rows).set_index("combo").sort_values("roc_auc", ascending=False)


def model_correlation(predictions: pd.DataFrame) -> pd.DataFrame:
    """3x3 Pearson correlation between the base models' probabilities."""
    return predictions[["gb_prob", "mlp_prob", "cnn_prob"]].corr()


def error_overlap(predictions: pd.DataFrame) -> pd.DataFrame:
    """Count buildings by how many of the three base models misclassify
    them at DECISION_THRESHOLD. Returns counts for 0 through 3."""
    y_true = predictions["true_label"].to_numpy()
    wrong = np.zeros(len(predictions), dtype=int)

    for column in ("gb_prob", "mlp_prob", "cnn_prob"):
        pred = (predictions[column].to_numpy() >= DECISION_THRESHOLD).astype(int)
        wrong += (pred != y_true).astype(int)

    labels = {0: "All three correct", 1: "One wrong",
              2: "Two wrong", 3: "All three wrong"}

    counts = pd.Series(wrong).value_counts().sort_index()

    return pd.DataFrame({
        "n_models_wrong": counts.index,
        "count": counts.to_numpy(),
        "description": [labels[k] for k in counts.index],
    })


def misclassification_by_heating_type(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per heating type: building count, misclassified count and rate, and
    false positive / false negative counts. Sorted by rate descending."""
    rows = []

    for heating_type, group in predictions.groupby("heating_type"):
        n_total = len(group)
        n_wrong = int((~group["correct"]).sum())

        rows.append({
            "heating_type": heating_type,
            "n_total": n_total,
            "n_misclassified": n_wrong,
            "misclass_rate_pct": round(100 * n_wrong / n_total, 1),
            "false_positive": int(((group["true_label"] == 0)
                                   & (group["pred_label"] == 1)).sum()),
            "false_negative": int(((group["true_label"] == 1)
                                   & (group["pred_label"] == 0)).sum()),
        })

    return (pd.DataFrame(rows)
            .sort_values("misclass_rate_pct", ascending=False)
            .reset_index(drop=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=str(WORK_DIR))
    parser.add_argument("--features", default=None)
    parser.add_argument("--profiles", default=None)
    parser.add_argument("--metadata", default=str(METADATA_CSV))
    parser.add_argument("--with-stacking", action="store_true",
                        help="Also fit a stacking meta-learner (about 6x slower)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    results_dir = Path(args.work_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    features_csv = args.features or (results_dir / "dallas_features_groupA_E.csv")
    profiles_csv = args.profiles or (results_dir / "dallas_cnn_profiles.csv")

    print(f"Work folder: {results_dir}")
    print("=" * 60)
    print("Heat pump detection -- 5-fold cross-validation")
    print("=" * 60)

    data = build_dataset(features_csv, profiles_csv, args.metadata)

    if args.with_stacking:
        print("\nStacking enabled: this run takes roughly 6x longer.")

    predictions, _ = run_cross_validation(data, args.with_stacking, verbose)
    predictions.to_csv(results_dir / "predictions.csv", index=False)

    per_fold, summary = summarize_folds(predictions)
    per_fold.to_csv(results_dir / "metrics_per_fold.csv")
    summary.to_csv(results_dir / "metrics_summary.csv")

    print("\n" + "=" * 60)
    print("Ensemble performance (mean +/- std across folds)")
    print("=" * 60)
    for metric, row in summary.iterrows():
        print(f"  {metric:<10}: {row['mean']:.4f} +/- {row['std']:.4f}")

    comparison = model_comparison(predictions)
    comparison.to_csv(results_dir / "model_comparison.csv")
    print("\nModel comparison:")
    print(comparison[["accuracy", "roc_auc"]].round(4).to_string())

    confusion = pooled_confusion(predictions)
    confusion.to_csv(results_dir / "confusion_matrix.csv")
    print("\nConfusion matrix:")
    print(confusion.to_string())

    ablation = model_ablation(predictions)
    ablation.to_csv(results_dir / "model_ablation.csv")
    print("\nModel ablation:")
    print(ablation[["n_models", "accuracy", "roc_auc"]].round(4).to_string())

    correlation = model_correlation(predictions)
    correlation.to_csv(results_dir / "model_correlation.csv")
    print("\nPairwise prediction correlation:")
    print(correlation.round(3).to_string())

    overlap = error_overlap(predictions)
    overlap.to_csv(results_dir / "error_overlap.csv", index=False)
    print("\nError overlap:")
    print(overlap.to_string(index=False))

    by_type = misclassification_by_heating_type(predictions)
    by_type.to_csv(results_dir / "misclassification_by_type.csv", index=False)
    print("\nMisclassification by heating system:")
    print(by_type.to_string(index=False))

    with open(results_dir / "run_summary.json", "w") as handle:
        json.dump({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "random_state": RANDOM_STATE,
            "n_folds": N_FOLDS,
            "n_buildings": int(len(data["y"])),
            "n_features": len(data["stat_columns"]),
            "positive_rate": round(float(data["y"].mean()), 4),
            "decision_threshold": DECISION_THRESHOLD,
            "stacking_enabled": bool(args.with_stacking),
            "ensemble_metrics_mean": {m: round(float(r["mean"]), 4)
                                      for m, r in summary.iterrows()},
            "ensemble_metrics_std": {m: round(float(r["std"]), 4)
                                     for m, r in summary.iterrows()},
        }, handle, indent=2)

    print(f"\nAll outputs written to {results_dir}")


if __name__ == "__main__":
    main()
