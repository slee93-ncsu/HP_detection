"""
Load the trained bundle and run the soft-voting ensemble.

The bundle written by 2. Reference/1. Code/05_train_final_model.py
holds sklearn objects and the HybridCNN weights as numpy arrays. Only the
HybridCNN class definition is needed here; it must match the one in
2. Reference/1. Code/03_train_evaluate.py layer for layer.

Preprocessing mirrors training:
    - statistical columns reordered to bundle["stat_columns"]
    - NaN and +/-inf replaced with 0
    - Gradient Boosting gets raw features; MLP and CNN get scaled features
    - probabilities of the three models are averaged
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn

from .profiles import PROFILE_LENGTH, PROFILE_NAMES, profile_columns

BUNDLE_FORMAT = "HP_detection_model_bundle"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "hp_detection_model.joblib"
N_CHANNELS = len(PROFILE_NAMES)
CNN_DROPOUT = 0.2  # inactive in eval mode; kept so the layers match training

# Decision threshold for hp_predicted. The model file stores the 0.5 used in
# cross-validation; 0.35 is used here because, on ResStock buildings in the
# counties adjacent to Dallas County, 0.5 misses a large share of heat pumps
# while 0.35 keeps Dallas County performance unchanged (F1 0.843 -> 0.848).
# See 2. Reference/2. Results/threshold_comparison.csv.
DEFAULT_THRESHOLD = 0.35


class HybridCNN(nn.Module):
    """Same architecture as 2. Reference/1. Code/03_train_evaluate.py."""

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


def load_bundle(path=None) -> dict:
    """Load the model bundle. Only load bundles from trusted sources:
    joblib files can execute code when loaded."""
    path = Path(path) if path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")

    bundle = joblib.load(path)

    if bundle.get("bundle_format") != BUNDLE_FORMAT:
        raise ValueError(f"{path} is not an HP_detection model bundle.")

    trained_with = bundle.get("software", {}).get("scikit_learn")
    if trained_with and trained_with != sklearn.__version__:
        warnings.warn(
            f"Model was trained with scikit-learn {trained_with}, running "
            f"{sklearn.__version__}. Install the pinned version for "
            "identical predictions.", stacklevel=2)

    if tuple(bundle["profile_names"]) != PROFILE_NAMES:
        raise ValueError("Bundle profile order does not match this package.")

    cnn = HybridCNN(bundle["cnn_n_stat_features"])
    cnn.load_state_dict({k: torch.from_numpy(np.array(v, copy=True))
                         for k, v in bundle["cnn_state_dict"].items()})
    cnn.eval()
    bundle["_cnn"] = cnn

    return bundle


def prepare_inputs(bundle: dict, features: pd.DataFrame, profiles: pd.DataFrame):
    """Return (x_stat, x_profiles) arrays in the order the model expects."""
    missing = [c for c in bundle["stat_columns"] if c not in features.columns]
    if missing:
        raise ValueError(f"Missing feature columns, e.g. {missing[:5]}")

    x_stat = (features[bundle["stat_columns"]]
              .replace([np.inf, -np.inf], np.nan).fillna(0.0)
              .to_numpy(dtype=np.float32, copy=True))

    x_prof = (profiles[profile_columns()]
              .replace([np.inf, -np.inf], np.nan).fillna(0.0)
              .to_numpy(dtype=np.float32, copy=True)
              .reshape(-1, N_CHANNELS, PROFILE_LENGTH))

    return x_stat, x_prof


def predict(bundle: dict, features: pd.DataFrame, profiles: pd.DataFrame,
            threshold: Optional[float] = None) -> pd.DataFrame:
    """Score buildings. `features` and `profiles` share the same index
    (building id) and row order. `threshold` defaults to DEFAULT_THRESHOLD."""
    threshold = DEFAULT_THRESHOLD if threshold is None else float(threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    if not features.index.equals(profiles.index):
        raise ValueError("features and profiles must have the same index.")

    x_stat, x_prof = prepare_inputs(bundle, features, profiles)
    x_scaled = bundle["scaler"].transform(x_stat).astype(np.float32)

    p_gb = bundle["gradient_boosting"].predict_proba(x_stat)[:, 1]
    p_mlp = bundle["mlp"].predict_proba(x_scaled)[:, 1]

    with torch.no_grad():
        logits = bundle["_cnn"](torch.from_numpy(x_prof), torch.from_numpy(x_scaled))
        p_cnn = torch.sigmoid(logits).numpy()

    prob = np.mean(np.vstack([p_gb, p_mlp, p_cnn]), axis=0)

    return pd.DataFrame({
        "hp_probability": prob,
        "hp_predicted": (prob >= threshold).astype(int),
        "p_gradient_boosting": p_gb,
        "p_mlp": p_mlp,
        "p_cnn": p_cnn,
    }, index=features.index)
