"""Turn a Strava activities export into the model input row (tasks A1 and A2).

Two steps, both in this file:

1. clean_activities(path): the raw export has duplicate column names, text-formatted
   numbers and activity types the model does not use; this returns a clean activity log.
2. build_features(activities, event_day): takes that clean log and returns exactly the
   columns the model was trained on, in the order of FEATURE_CONTRACT.md.

Typical use from the API:

    from RunGuard.ml_logic.strava import build_features, clean_activities

    activities = clean_activities("activities.csv")
    row = build_features(activities)  # event day = the day after the last activity

Conversion rules: FEATURE_CONTRACT.md and docs/STRAVA_CONVERSION_DECISIONS.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# --- Rules --------------------------------------------------------------------------------------

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

WINDOW_DAYS = 7     # contract section 2: the 7 full days before the event day
MAX_SESSIONS = 2    # contract: "nr. sessions" is capped at the training-data maximum

# Decision 4: a multi-day activity is spread over its elapsed time, unless the recording
# was probably left running (elapsed time more than this many times the moving time)
MAX_ELAPSED_TO_MOVING = 2

# Contract section 1: the 4 base features, in contract order
BASE_FEATURES = ["nr. sessions", "total km", "strength training", "hours alternative"]

# Contract section 2: no suffix = 7 days before the event, ".1" = 6 days before, ..., ".6" = 1 day before
SUFFIXES = [""]
for k in range(1, WINDOW_DAYS):
    SUFFIXES.append("." + str(k))

# Day by day, oldest first, the 4 features of each day: "nr. sessions", ..., "hours alternative.6"
CONTRACT_COLUMNS = []
for suffix in SUFFIXES:
    for feature in BASE_FEATURES:
        CONTRACT_COLUMNS.append(feature + suffix)


# --- Step 1: clean activity log (A1) --------------------------------------------------------------

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


# --- Step 2: model input row (A2) ---------------------------------------------------------------

def build_features(activities: pd.DataFrame, event_day=None) -> pd.DataFrame:
    """Return the model input for one event day: a table with 1 row and the 28 contract columns.

    `activities` is the clean activity log from `clean_activities`. `event_day` is a date
    (string like "2026-10-15" or a Timestamp); by default it is the day after the last
    activity, as defined in the contract. The window is the 7 full days before it.
    """
    if activities.empty:
        if event_day is None:
            raise ValueError("no activities and no event_day: cannot tell which week to describe")
        # No training at all: every feature is 0 ("If no activity that day: 0")
        event_day = pd.Timestamp(event_day).normalize()
        return pd.DataFrame(0, index=pd.DatetimeIndex([event_day], name="event day"),
                            columns=CONTRACT_COLUMNS)

    daily = daily_features(activities)
    if event_day is None:
        # pd.Timedelta is a length of time that can be added to or subtracted from a date
        event_day = daily.index[-1] + pd.Timedelta(days=1)

    # pd.Timestamp(...) turns a string like "2026-10-15" into a date; normalize() drops the time
    return contract_rows(daily, [pd.Timestamp(event_day).normalize()])


def split_by_day(activities: pd.DataFrame) -> pd.DataFrame:
    """Kilometres and moving time per activity and day (decision 4).

    A multi-day activity is spread evenly over the time it covers: each day gets
    the share of that time that falls inside it, and the same share of the moving
    time and of the distance. Breaks such as sleep are spread with it.

    The covered time is the elapsed time, or the moving time if the recording was
    probably left running (elapsed time more than MAX_ELAPSED_TO_MOVING times the moving time).
    """
    parts = []

    # itertuples() goes through the table row by row; each row's columns are attributes,
    # e.g. activity.start or activity.km
    for activity in activities.itertuples():
        if activity.elapsed_s <= MAX_ELAPSED_TO_MOVING * activity.moving_s:
            covered_s = activity.elapsed_s
        else:
            covered_s = activity.moving_s

        start = activity.start
        end = start + pd.Timedelta(seconds=covered_s)
        midnight_after_start = start.normalize() + pd.Timedelta(days=1)

        if covered_s == 0 or end <= midnight_after_start:
            # Ends on its start day: nothing to split
            parts.append({"day": activity.day, "group": activity.group,
                          "km": activity.km, "moving_s": activity.moving_s})
            continue

        day = start.normalize()
        while day < end:
            next_day = day + pd.Timedelta(days=1)
            # The part of [start, end] inside [day, next_day], as a share of the whole
            share = (min(end, next_day) - max(start, day)).total_seconds() / covered_s
            parts.append({"day": day, "group": activity.group,
                          "km": activity.km * share, "moving_s": activity.moving_s * share})
            day = next_day

    # A list of dictionaries becomes a table: one row per dictionary, one column per key
    return pd.DataFrame(parts)


def daily_features(activities: pd.DataFrame) -> pd.DataFrame:
    """The 4 base features for every calendar day, rest days included (all zeros)."""
    # Sessions and strength trainings are counted once, on the day the activity starts
    strength = activities[activities["group"] == "strength"]

    # Distance and hours are split across the days an activity covers
    parts = split_by_day(activities)
    run_parts = parts[parts["group"] == "run"]
    alternative_parts = parts[parts["group"] == "alternative"]

    # Every day from the first activity to the last day an activity reaches, one step per day
    days = pd.date_range(activities["day"].min(), parts["day"].max(), freq="D")

    def per_day(series):
        # Counts and sums below only contain days with activities;
        # reindex puts them on the full calendar and fills the missing (rest) days with 0
        return series.reindex(days, fill_value=0)

    daily = pd.DataFrame(index=days)
    # groupby("day").size() counts rows (activities) per day;
    # clip(upper=...) replaces every value above the limit with the limit
    daily["nr. sessions"] = per_day(activities.groupby("day").size()).clip(upper=MAX_SESSIONS).astype(int)
    daily["total km"] = per_day(run_parts.groupby("day")["km"].sum()).astype(float).round(2)
    daily["strength training"] = per_day(strength.groupby("day").size()).astype(int)
    daily["hours alternative"] = (per_day(alternative_parts.groupby("day")["moving_s"].sum()) / 3600).round(2)
    daily.index.name = "day"
    return daily


def contract_rows(daily: pd.DataFrame, event_days=None) -> pd.DataFrame:
    """One row of the 28 contract columns per event day, from the 7 full days before it.

    By default: every day with a complete 7-day history, plus the day after the last
    activity. `build_features` uses this for a single event day; the full history is
    useful for exploring the data.
    """
    if event_days is None:
        next_day = daily.index[-1] + pd.Timedelta(days=1)
        event_days = daily.index[WINDOW_DAYS:].append(pd.DatetimeIndex([next_day]))
    event_days = pd.DatetimeIndex(event_days)

    # A calendar wide enough for every window: days before the first activity or
    # after the last one have no activities, so they are rest days (all zeros)
    first_day = min(daily.index[0], event_days.min() - pd.Timedelta(days=WINDOW_DAYS))
    calendar = pd.date_range(first_day, event_days.max(), freq="D")
    full = daily.reindex(calendar, fill_value=0)

    parts = []
    # enumerate gives (position, value) pairs: (0, ""), (1, ".1"), ..., (6, ".6")
    for k, suffix in enumerate(SUFFIXES):
        days_before = WINDOW_DAYS - k  # "" -> 7 days before, ".6" -> 1 day before
        # Subtracting a Timedelta from all event days at once gives the day to read for each;
        # reindex then picks those days' rows from the calendar
        part = full.reindex(event_days - pd.Timedelta(days=days_before))[BASE_FEATURES]
        part.columns = [feature + suffix for feature in BASE_FEATURES]
        # Label the rows by event day, so the 7 parts line up when glued together
        part.index = event_days
        parts.append(part)

    # pd.concat(..., axis=1) puts the 7 parts side by side, matching rows by event day
    rows = pd.concat(parts, axis=1)[CONTRACT_COLUMNS]
    rows.index.name = "event day"
    return rows
