"""
Compute permutation importance for the soft-voting ensemble.

Importance is the drop in ensemble ROC-AUC when one input is shuffled
within a fold's test set, averaged over --n-repeats shuffles and over the
five folds.

Inputs: same three CSVs as script 03 (features, profiles, metadata).
Output: permutation_importance.csv  one row per input, sorted by drop
        group_importance.csv        the 102 features aggregated by the
                                    A-E prefix of their name

Depends on 03_train_evaluate.py, which must sit in the same folder. Its
filename starts with a digit so it is loaded by path rather than
imported. Cross-validation is re-run with the same seed to obtain the
fold models, then those fitted models are reused for every shuffle.

The 102 statistical features are shuffled one column at a time. The four
profiles are shuffled one channel at a time, moving all 24 hours of a
building together so each building's daily shape stays intact.

Usage:
    python 04_permutation_importance.py --metadata "path/to/TX_upgrade0.csv"
    python 04_permutation_importance.py --metadata "..." --n-repeats 2
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

WORK_DIR = Path(os.environ.get("HP_WORK_DIR", "outputs"))

METADATA_CSV = Path(os.environ.get(
    "HP_METADATA_CSV",
    str(Path(os.environ.get("HP_DATA_ROOT", "data")) / "TX_upgrade0.csv"),
))

N_REPEATS = 5
RANDOM_STATE = 42

TRAIN_SCRIPT = Path(__file__).resolve().parent / "03_train_evaluate.py"


def load_training_module():
    """Load 03_train_evaluate.py by path and return it as a module.

    Provides build_dataset, run_cross_validation, predict_cnn, soft_vote,
    get_device, and PROFILE_NAMES.
    """
    if not TRAIN_SCRIPT.exists():
        raise SystemExit(f"Cannot find {TRAIN_SCRIPT}. Keep all four scripts "
                         "in the same folder.")

    spec = importlib.util.spec_from_file_location("train_evaluate", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_evaluate"] = module
    spec.loader.exec_module(module)

    return module


def ensemble_probability(artifacts, x_stat, x_profiles, device, tr) -> np.ndarray:
    """Score one fold's test set with that fold's three fitted models.

    x_stat and x_profiles may be permuted copies. The fold's scaler is
    applied to x_stat before the MLP and CNN see it.
    """
    x_stat_scaled = artifacts["scaler"].transform(x_stat)

    return tr.soft_vote(
        artifacts["gb_model"].predict_proba(x_stat)[:, 1],
        artifacts["mlp_model"].predict_proba(x_stat_scaled)[:, 1],
        tr.predict_cnn(artifacts["cnn_model"], x_profiles, x_stat_scaled, device),
    )


def permutation_importance(data, artifacts_list, tr, n_repeats=N_REPEATS,
                           verbose=True) -> pd.DataFrame:
    """Return a DataFrame with columns kind, feature, auc_drop_mean,
    auc_drop_std, sorted by auc_drop_mean descending.

    kind is "stat" for the 102 features or "profile" for the 4 channels.
    auc_drop_std is the spread across folds, not across repeats.
    """
    device = tr.get_device()
    rng = np.random.default_rng(RANDOM_STATE)

    items = [("stat", name, i) for i, name in enumerate(data["stat_columns"])]
    items += [("profile", name, i) for i, name in enumerate(tr.PROFILE_NAMES)]

    records = {(kind, name): [] for kind, name, _ in items}

    for artifacts in artifacts_list:
        test_idx = artifacts["test_index"]

        x_stat = data["x_stat"][test_idx].copy()
        x_profiles = data["x_profiles"][test_idx].copy()
        y_test = data["y"][test_idx]

        baseline = roc_auc_score(y_test, artifacts["ensemble_prob"])

        if verbose:
            print(f"\nFold {artifacts['fold']}: baseline ROC-AUC = {baseline:.4f}")

        for position, (kind, name, index) in enumerate(items, start=1):
            drops = []

            for _ in range(n_repeats):
                if kind == "stat":
                    permuted_stat = x_stat.copy()
                    permuted_stat[:, index] = rng.permutation(permuted_stat[:, index])
                    permuted_profiles = x_profiles
                else:
                    permuted_stat = x_stat
                    permuted_profiles = x_profiles.copy()
                    # axis=0 shuffles buildings, keeping each 24-hour row intact.
                    permuted_profiles[:, index, :] = rng.permutation(
                        permuted_profiles[:, index, :], axis=0)

                probability = ensemble_probability(
                    artifacts, permuted_stat, permuted_profiles, device, tr)

                drops.append(baseline - roc_auc_score(y_test, probability))

            records[(kind, name)].append(float(np.mean(drops)))

            if verbose and position % 20 == 0:
                print(f"    {position}/{len(items)} done")

    rows = [{"kind": kind, "feature": name,
             "auc_drop_mean": float(np.mean(values)),
             "auc_drop_std": float(np.std(values))}
            for (kind, name), values in records.items()]

    return (pd.DataFrame(rows)
            .sort_values("auc_drop_mean", ascending=False)
            .reset_index(drop=True))


def group_importance(importance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the "stat" rows by the first character of the feature name
    (A-E). Returns total_auc_drop, mean_auc_drop, and n_features per group."""
    stat_only = importance[importance["kind"] == "stat"].copy()
    stat_only["group"] = stat_only["feature"].str[0]

    return (stat_only.groupby("group")["auc_drop_mean"]
            .agg(["sum", "mean", "count"])
            .rename(columns={"sum": "total_auc_drop", "mean": "mean_auc_drop",
                             "count": "n_features"})
            .sort_values("total_auc_drop", ascending=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default=str(WORK_DIR))
    parser.add_argument("--features", default=None)
    parser.add_argument("--profiles", default=None)
    parser.add_argument("--metadata", default=str(METADATA_CSV))
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    results_dir = Path(args.work_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    features_csv = args.features or (results_dir / "dallas_features_groupA_E.csv")
    profiles_csv = args.profiles or (results_dir / "dallas_cnn_profiles.csv")

    tr = load_training_module()

    print(f"Work folder: {results_dir}")
    print("=" * 60)
    print("Permutation importance")
    print("=" * 60)

    data = tr.build_dataset(features_csv, profiles_csv, args.metadata)

    print("\nRe-running cross-validation to obtain fold models...")
    _, artifacts_list = tr.run_cross_validation(data, verbose=verbose)

    print("\n" + "=" * 60)
    print("Permuting inputs")
    print("=" * 60)

    importance = permutation_importance(data, artifacts_list, tr,
                                        args.n_repeats, verbose)
    importance.to_csv(results_dir / "permutation_importance.csv", index=False)

    groups = group_importance(importance)
    groups.to_csv(results_dir / "group_importance.csv")

    print("\nTop 20:")
    print(importance.head(20).to_string(
        index=False, columns=["feature", "kind", "auc_drop_mean", "auc_drop_std"]))

    print("\nBy feature group:")
    print(groups.to_string())

    print(f"\nAll outputs written to {results_dir}")


if __name__ == "__main__":
    main()
