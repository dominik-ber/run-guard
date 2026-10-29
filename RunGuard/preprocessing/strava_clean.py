"""Clean a Strava activities export into one tidy row per activity (task A1).

The raw export has duplicate column names, text-formatted numbers and activity
types the model does not use. This module turns it into a clean activity log
that `RunGuard.features.strava` can build the contract features from.

Conversion rules: FEATURE_CONTRACT.md and docs/STRAVA_CONVERSION_DECISIONS.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Contract section 3: which Strava activity types feed which feature.
# Trail Run and Virtual Run (treadmill) count as runs, as listed in the contract.
RUN_TYPES = {"Run", "Trail Run", "Virtual Run"}
STRENGTH_TYPES = {"Workout", "Weight Training", "Crossfit"}
ALTERNATIVE_TYPES = {"Ride", "Virtual Ride", "Mountain Bike Ride", "Gravel Ride", "Swim",
                     "Open Water Swim", "Hike", "Walk", "Yoga", "Pilates"}

# Decision 2: runs shorter than this (separately recorded warm-ups and cool-downs) are dropped
MIN_RUN_KM = 2

# Strava writes dates like "Oct 14, 2026, 9:30:23 AM"
STRAVA_DATE_FORMAT = "%b %d, %Y, %I:%M:%S %p"


def clean_activities(path) -> pd.DataFrame:
    """Read a Strava `activities.csv` export and return the clean activity log.

    One row per activity the contract uses, sorted by start time, with columns:

    - start:     start time, UTC as exported (decision 1)
    - day:       start date (UTC), time set to midnight
    - type:      Strava "Activity Type"
    - group:     "run", "strength" or "alternative" (contract section 3)
    - km:        distance in kilometres (decision 3)
    - moving_s:  moving time in seconds
    - elapsed_s: elapsed time in seconds, breaks included
    """
    # low_memory=False: read the whole file before guessing each column's type,
    # so mixed columns do not trigger a DtypeWarning
    raw = pd.read_csv(Path(path), low_memory=False)
    return prepare_activities(raw)


def prepare_activities(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean an already loaded export (same output as `clean_activities`)."""
    activities = pd.DataFrame({
        "start": pd.to_datetime(raw["Activity Date"], format=STRAVA_DATE_FORMAT),
        "type": raw["Activity Type"],
        # The CSV has two "Distance" columns; pandas renames the second one to "Distance.1".
        # It is always in metres, while the first one mixes km and metres (decision 3).
        "km": raw["Distance.1"] / 1000,
        # pd.to_numeric(..., errors="coerce") turns anything that is not a number into NaN
        # instead of raising an error; .fillna(0) then replaces those NaN with 0
        "elapsed_s": pd.to_numeric(raw["Elapsed Time"], errors="coerce").fillna(0),
        "moving_s": pd.to_numeric(raw["Moving Time"], errors="coerce").fillna(0),
    })

    # np.select(conditions, choices, default) works like if / elif / else on every row:
    # each row gets the choice of the first condition that is True for it
    activities["group"] = np.select(
        [activities["type"].isin(RUN_TYPES),
         activities["type"].isin(STRENGTH_TYPES),
         activities["type"].isin(ALTERNATIVE_TYPES)],
        ["run", "strength", "alternative"],
        default="excluded",
    )

    # Keep rows that are in a contract group and are not short runs (decision 2).
    # "&" means "and", "~" means "not"; both work row by row on True/False columns.
    short_run = (activities["group"] == "run") & (activities["km"] < MIN_RUN_KM)
    activities = activities[(activities["group"] != "excluded") & ~short_run]

    # reset_index(drop=True) renumbers the rows 0, 1, 2, ... after filtering and sorting
    activities = activities.sort_values("start").reset_index(drop=True)

    # .dt gives access to date methods; normalize() keeps the date and sets the time to 00:00
    activities["day"] = activities["start"].dt.normalize()

    return activities[["start", "day", "type", "group", "km", "moving_s", "elapsed_s"]]
