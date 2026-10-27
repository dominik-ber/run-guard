# RunGuard Feature Contract v0 (using day_approach dataset)

This file defines the exact input the ML model expects for training & prediction. The following must produce identical columns:

- **Training side (D2):** selects these columns from `day_approach_maskedID_timeseries.csv`.
- **App side (A2):** builds one row with these columns from a runner's Strava `activities.csv`.

Copy column names from this file. Don't retype them: `nr. sessions` with a missing space or dot breaks the model.

**Note: All features below are raw values, without any scaling or normalization.** Scaling, if needed, belongs in the model pipeline, so both the training side and the app side must supply the unscaled values exactly as defined here.

## 1. Base features

Each base feature is computed once per day for 7 days (see section 2), giving **4 × 7 = 28 columns**.

| # | Column (exact name) | Unit | How it's computed from Strava | If no activity that day |
|---|---|---|---|---|
| 1 | `nr. sessions` | count | Number of activities that day, counting runs, strength and alternative training (section 3). **Capped at 2** (the training-data maximum). | 0 |
| 2 | `total km` | km | Sum of `Distance` of run activities that day. `Distance` is in km; the duplicate `Distance.1` column is in metres. | 0 |
| 3 | `strength training` | count | Number of strength activities that day. | 0 |
| 4 | `hours alternative` | hours | Sum of `Moving Time` of alternative activities that day. `Moving Time` is in seconds, so divide by 3600. | 0 |

## 2. Time window and column order

- **Event day = date after last 7 days of data available** (assuming the upload might not be done using today's data).
- **The time window is the 7 full days before an event.** In the training data the event day itself is not part of the window.
- **Suffixes count up toward the event day**:

| Suffix | Day |
|---|---|
| *(none)* | 7 days before event |
| `.1` | 6 days before |
| `.2` | 5 days before |
| `.3` | 4 days before |
| `.4` | 3 days before |
| `.5` | 2 days before |
| `.6` | 1 day before |

**Column order matters,** because the model reads columns by position. The order is day by day, oldest first, with the 4 features in table order within each day.

**The input for model training & prediction should follow exactly the below column order:**

```
nr. sessions
total km
strength training
hours alternative
nr. sessions.1
total km.1
strength training.1
hours alternative.1
nr. sessions.2
total km.2
strength training.2
hours alternative.2
nr. sessions.3
total km.3
strength training.3
hours alternative.3
nr. sessions.4
total km.4
strength training.4
hours alternative.4
nr. sessions.5
total km.5
strength training.5
hours alternative.5
nr. sessions.6
total km.6
strength training.6
hours alternative.6
```

## 3. Strava activity type mapping

Based on the `Activity Type` column in Strava data, this table shows which activities feed which features.

| Group | Activity types | Feeds feature |
|---|---|---|
| Strength | Workout, Weight Training, Crossfit | `strength training`, `nr. sessions` |
| Alternative | Ride, Virtual Ride, Mountain Bike Ride, Gravel Ride, Swim, Open Water Swim, Hike, Walk, Yoga, Pilates | `hours alternative`, `nr. sessions` |
| Run | Run, Trail Run, Virtual Run | `total km`, `nr. sessions` |
| Excluded | Any other type not listed above | nothing |

## 4. Features left out of v0

| Training feature | Reason |
|---|---|
| `km Z3-4`, `km Z5-T1-T2` | They need a heart-rate zone approximation, and only about 53% of my runs have HR. Let's do this as in v1 iteration. |
| `km sprinting` | No Strava source. |
| `perceived exertion` | Strava RPE is filled on 0.5% of runs, and its scale (1–10) differs from the training data (0–1). |
| `perceived trainingSuccess`, `perceived recovery` | No Strava source. |

##  5. Target and non-feature columns

These columns exist only in the training data. The app never builds them.

| Column | Role | Definition |
|---|---|---|
| `injury` | Target | 1 = the athlete got injured on the event day, 0 = no injury. 583 of 42,766 rows are 1 (1.36%), spread over 63 of 74 athletes. |
| `Athlete ID` | Split only | Used for the athlete-grouped train/test split. **Never a feature.** |
| `Date` | Split / ordering only | Event day, counted in days from the first record (0–2,673). Consecutive rows of one athlete share 6 of 7 window days. **Never a feature.** |

## 5. Example

[`example_row.csv`](https://drive.google.com/file/d/1h3a4XUSqcwW2y8qOtnYyZA0p0gchx4uB/view?usp=drive_link) is a real row built from Dominik's export with event day 2026-10-15.
Use it as a mock input for the model training, API, UI etc. before Strava data processing pipe is ready.

*Contract version: v0*
