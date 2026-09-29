"""
Four 24-hour load profiles per building (CNN input).

Copied from 2. Code/02_build_profiles.py without changes to the profile
logic. Column order is profile-major and must match the trained model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PROFILE_NAMES = ("winter_weekday", "summer_weekday", "shoulder_weekday", "peak_day")
PROFILE_LENGTH = 24

SEASON_MAP = {
    12: "winter", 1: "winter", 2: "winter",
    3: "shoulder", 4: "shoulder", 5: "shoulder",
    6: "summer", 7: "summer", 8: "summer",
    9: "shoulder", 10: "shoulder", 11: "shoulder",
}


def profile_columns() -> list:
    """The 96 column names in profile-major order."""
    return [f"{name}_h{h:02d}" for name in PROFILE_NAMES for h in range(PROFILE_LENGTH)]


def average_24h(subset: pd.DataFrame) -> list:
    """Mean load per hour of day over the given rows (24 floats)."""
    if len(subset) == 0:
        return [np.nan] * PROFILE_LENGTH

    profile = subset.groupby("hour")["load"].mean()

    return [profile.get(h, np.nan) for h in range(PROFILE_LENGTH)]


def compute_profiles(hourly: pd.DataFrame) -> dict:
    """Return {"{profile}_h{hour:02d}": value} for the 96 profile values."""
    frame = hourly.copy()
    frame["is_weekday"] = frame.index.weekday < 5  # 0=Mon .. 6=Sun
    frame["season"] = frame["month"].map(SEASON_MAP)

    daily_total = frame.groupby("date")["load"].sum()

    if len(daily_total) > 0:
        peak_day = frame[frame["date"] == daily_total.idxmax()]
        peak_profile = (peak_day.set_index("hour")["load"]
                        .reindex(range(PROFILE_LENGTH)).tolist())
    else:
        peak_profile = [np.nan] * PROFILE_LENGTH

    def seasonal(season):
        return average_24h(frame[(frame["season"] == season) & frame["is_weekday"]])

    profiles = {
        "winter_weekday": seasonal("winter"),
        "summer_weekday": seasonal("summer"),
        "shoulder_weekday": seasonal("shoulder"),
        "peak_day": peak_profile,
    }

    return {f"{name}_h{hour:02d}": value
            for name in PROFILE_NAMES
            for hour, value in enumerate(profiles[name])}
