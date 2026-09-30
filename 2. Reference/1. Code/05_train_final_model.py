"""
Train the final heat-pump detection ensemble on all available development data
and save it as one model bundle.

The bundle contains the fitted Gradient Boosting model, fitted MLP model,
fitted StandardScaler, HybridCNN learned parameters, feature/profile order,
decision threshold, and basic metadata.

03_train_evaluate.py must be in the same folder.
"""

from __future__ import annotations

import argparse
import importlib.util
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


TRAIN_SCRIPT = Path(__file__).resolve().parent / "03_train_evaluate.py"


def load_training_module():
    if not TRAIN_SCRIPT.exists():
        raise SystemExit(
            f"Cannot find {TRAIN_SCRIPT}. Keep 05_train_final_model.py "
            "in the same folder as 03_train_evaluate.py."
        )

    spec = importlib.util.spec_from_file_location("train_evaluate", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["train_evaluate"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="outputs")
    parser.add_argument("--features", default=None)
    parser.add_argument("--profiles", default=None)
    parser.add_argument("--metadata", required=True)
    parser.add_argument(
        "--model-out",
        default=str(Path("model") / "hp_detection_model.joblib"),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    tr = load_training_module()
    verbose = not args.quiet

    work_dir = Path(args.work_dir)
    features_csv = args.features or (work_dir / "dallas_features_groupA_E.csv")
    profiles_csv = args.profiles or (work_dir / "dallas_cnn_profiles.csv")

    print("=" * 60)
    print("Final heat-pump detection model training")
    print("=" * 60)

    data = tr.build_dataset(
        features_csv,
        profiles_csv,
        args.metadata,
        verbose=verbose,
    )

    tr.set_seeds()
    device = tr.get_device()
    print(f"\nDevice: {device}")
    print(f"Training buildings: {len(data['y']):,}")

    scaler = StandardScaler().fit(data["x_stat"])
    x_stat_scaled = scaler.transform(data["x_stat"])

    print("\nTraining Gradient Boosting on all buildings...")
    gb = GradientBoostingClassifier(**tr.GB_PARAMS).fit(
        data["x_stat"], data["y"]
    )

    print("Training MLP on all buildings...")
    mlp = MLPClassifier(**tr.MLP_PARAMS).fit(
        x_stat_scaled, data["y"]
    )

    print("Training HybridCNN on all buildings...")
    cnn = tr.train_cnn(
        data["x_profiles"],
        x_stat_scaled,
        data["y"].astype(np.float32),
        device,
        verbose=verbose,
    )

    cnn_state = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in cnn.state_dict().items()
    }

    bundle = {
        "bundle_format": "HP_detection_model_bundle",
        "bundle_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_scope": "Full Dallas County development dataset",
        "n_training_buildings": int(len(data["y"])),
        "positive_rate": float(data["y"].mean()),
        "decision_threshold": float(tr.DECISION_THRESHOLD),
        "positive_heating_types": sorted(tr.HP_HEATING_TYPES),
        "stat_columns": list(data["stat_columns"]),
        "profile_names": list(tr.PROFILE_NAMES),
        "profile_length": int(tr.PROFILE_LENGTH),
        "n_channels": int(tr.N_CHANNELS),
        "random_state": int(tr.RANDOM_STATE),
        "gradient_boosting": gb,
        "mlp": mlp,
        "scaler": scaler,
        "cnn_state_dict": cnn_state,
        "cnn_n_stat_features": int(data["x_stat"].shape[1]),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
    }

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_out, compress=3)

    print(f"\nSaved model bundle: {model_out}")
    print(f"Bundle size: {model_out.stat().st_size / (1024**2):.2f} MB")
    print(
        "\nThis final model is fitted on all development data. "
        "Use the cross-validation results from 03_train_evaluate.py "
        "for performance reporting."
    )


if __name__ == "__main__":
    main()
