"""
Command line entry point.

    hp-detect predict meter_data.csv --config mapping.yaml --out predictions.csv
    hp-detect predict path/to/resstock_parquet_dir --out predictions.csv
    hp-detect info
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from . import __version__
from .io import InputConfig

# Columns written by default; --details writes every column.
MAIN_COLUMNS = ["hp_probability", "hp_predicted", "quality_flag", "quality_notes"]


def _read_config(path) -> dict:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)

    import yaml  # only needed for YAML configs
    return yaml.safe_load(text) or {}


def _build_config(args) -> InputConfig:
    values = _read_config(args.config) if args.config else {}

    # A relative weather_file in the config file is relative to that file,
    # so the command works from any folder.
    weather = values.get("weather_file")
    if weather and not Path(weather).is_absolute():
        values["weather_file"] = str(Path(args.config).resolve().parent / weather)

    for f in fields(InputConfig):
        override = getattr(args, f.name, None)
        if override is not None:
            values[f.name] = override
    return InputConfig.from_dict(values)


def _add_input_options(parser) -> None:
    group = parser.add_argument_group("input layout (overrides the config file)")
    group.add_argument("--format", choices=["auto", "resstock", "long"])
    group.add_argument("--id-column", dest="id_column")
    group.add_argument("--timestamp-column", dest="timestamp_column")
    group.add_argument("--load-column", dest="load_column")
    group.add_argument("--load-unit", dest="load_unit", choices=["kWh", "Wh", "kW"])
    group.add_argument("--temp-column", dest="temp_column")
    group.add_argument("--temp-unit", dest="temp_unit", choices=["C", "F"])
    group.add_argument("--timestamp-convention", dest="timestamp_convention",
                       choices=["end", "start"])
    group.add_argument("--timezone")
    group.add_argument("--weather-file", dest="weather_file")
    group.add_argument("--weather-timestamp-column", dest="weather_timestamp_column")
    group.add_argument("--weather-temp-column", dest="weather_temp_column")
    group.add_argument("--weather-key-column", dest="weather_key_column")


def cmd_predict(args) -> None:
    from .model import DEFAULT_THRESHOLD, load_bundle, predict
    from .pipeline import build_inputs

    cfg = _build_config(args)
    bundle = load_bundle(args.model)
    features, profiles, checks = build_inputs(args.input, cfg, args.workers)
    result = predict(bundle, features, profiles, args.threshold).join(checks)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (result if args.details else result[MAIN_COLUMNS]).to_csv(out)

    if args.save_inputs:
        inputs_dir = Path(args.save_inputs)
        inputs_dir.mkdir(parents=True, exist_ok=True)
        features.to_csv(inputs_dir / "features.csv")
        profiles.to_csv(inputs_dir / "profiles.csv")

    n = len(result)
    n_hp = int(result["hp_predicted"].sum())
    n_low = int((result["quality_flag"] == "low").sum())
    print(f"\nDecision threshold   : {args.threshold if args.threshold is not None else DEFAULT_THRESHOLD}")
    print(f"Buildings scored     : {n:,}")
    print(f"Predicted heat pump  : {n_hp:,} ({100 * n_hp / n:.1f}%)")
    print(f"Low data quality     : {n_low:,}")
    print(f"Wrote {out}")


def cmd_info(args) -> None:
    from .model import DEFAULT_THRESHOLD, load_bundle

    bundle = load_bundle(args.model)
    keys = ("bundle_version", "created_utc", "training_scope",
            "n_training_buildings", "positive_rate", "decision_threshold",
            "positive_heating_types", "software")
    for key in keys:
        print(f"{key:22}: {bundle.get(key)}")
    print(f"{'n_stat_features':22}: {len(bundle['stat_columns'])}")
    print(f"{'package threshold':22}: {DEFAULT_THRESHOLD} (used for hp_predicted; "
          "decision_threshold above is the cross-validation value)")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="hp-detect", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="score buildings from meter data")
    p.add_argument("input", help="long-format CSV/parquet, or a ResStock parquet directory")
    p.add_argument("--config", help="YAML or JSON file describing the input layout")
    p.add_argument("--out", default="predictions.csv")
    p.add_argument("--model", default=None, help="model bundle (default: bundled model)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threshold", type=float, default=None,
                   help="decision threshold for hp_predicted (default 0.35)")
    p.add_argument("--details", action="store_true",
                   help="also write base-model probabilities and data checks")
    p.add_argument("--save-inputs", metavar="DIR",
                   help="also write the computed features and profiles")
    _add_input_options(p)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("info", help="show model bundle metadata")
    p.add_argument("--model", default=None)
    p.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
