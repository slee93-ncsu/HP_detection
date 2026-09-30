"""
Per-building data quality checks.

There is no ground truth at inference time, so the output carries these
checks alongside each prediction. A "low" flag does not block the
prediction; it marks buildings whose input differs from what the model
was trained on (a full, gap-free year of data).
"""

from __future__ import annotations

import pandas as pd

HOURS_PER_YEAR = 8760
MIN_COVERAGE_PCT = 90.0
MIN_SPAN_DAYS = 350


def assess(hourly: pd.DataFrame, raw_info: dict) -> dict:
    load = hourly["load"]
    temp = hourly["temp_c"]

    span_days = (hourly.index.max() - hourly.index.min()) / pd.Timedelta(days=1)
    load_cov = min(100.0, 100.0 * load.notna().sum() / HOURS_PER_YEAR)
    temp_cov = min(100.0, 100.0 * temp.notna().sum() / HOURS_PER_YEAR)
    n_months = hourly.loc[load.notna(), "month"].nunique()
    n_negative = int((load < 0).sum())

    problems = []
    if span_days < MIN_SPAN_DAYS:
        problems.append("less_than_one_year")
    if load_cov < MIN_COVERAGE_PCT:
        problems.append("load_gaps")
    if temp_cov < MIN_COVERAGE_PCT:
        problems.append("temperature_gaps")
    if n_months < 12:
        problems.append("missing_months")
    if n_negative:
        problems.append("negative_load")

    # Informational only: naive local timestamps repeat an hour at the DST
    # fall-back, which slightly inflates that one hour.
    notes = list(problems)
    if raw_info.get("n_duplicate_timestamps", 0):
        notes.append("duplicate_timestamps")
    if raw_info.get("trimmed"):
        notes.append("trimmed_to_last_12_months")

    return {
        "annual_kwh": float(load.sum()),
        "interval_minutes": raw_info.get("interval_minutes"),
        "period_start": hourly.index.min().strftime("%Y-%m-%d %H:%M"),
        "period_end": hourly.index.max().strftime("%Y-%m-%d %H:%M"),
        "span_days": round(float(span_days), 1),
        "load_coverage_pct": round(load_cov, 1),
        "temp_coverage_pct": round(temp_cov, 1),
        "quality_flag": "low" if problems else "ok",
        "quality_notes": ";".join(notes),
    }
