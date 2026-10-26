# RunGuard Feature Contract v0 (day approach)

This file defines the exact input the model expects. Two sides must produce identical columns:

- **Training side (D2):** selects these columns from `day_approach_maskedID_timeseries.csv` (Lövdal et al., 2021).
- **App side (A2):** builds one row with these columns from a runner's Strava `activities.csv`.

Copy column names from this file. Don't retype them: `nr. sessions` with a missing space or dot breaks the model.

## 1. Base features

Each base feature is computed once per day for 7 days (see section 2), giving **4 × 7 = 28 columns**.

| # | Column (exact name) | Unit | How it's computed from Strava | If no activity that day |
|---|---|---|---|---|
| 1 | `nr. sessions` | count | Number of activities that day, counting runs, strength and alternative training (section 3). **Capped at 2** (the training-data maximum). | 0 |
| 2 | `total km` | km | Sum of `Distance` of run activities that day. `Distance` is in km; the duplicate `Distance.1` column is in metres. | 0 |
| 3 | `strength training` | count | Number of strength activities that day. | 0 |
| 4 | `hours alternative` | hours | Sum of `Moving Time` of alternative activities that day. `Moving Time` is in seconds, so divide by 3600. | 0 |

No `-0.01` placeholder is needed. The training data uses it only in the perceived-feedback columns, and those are dropped (section 5).

## 2. Time window and column order

- **Event day = today**, using the **user's local date**.
- **The window is the 7 full days before today.** Today's activities are excluded, because in the training data the event day itself is not part of the window.
- **Suffixes count up toward the event day** (dataset README v2):

| Suffix | Day |
|---|---|
| *(none)* | 7 days before today (oldest) |
| `.1` | 6 days before |
| `.2` | 5 days before |
| `.3` | 4 days before |
| `.4` | 3 days before |
| `.5` | 2 days before |
| `.6` | yesterday |

- **Column order matters,** because the model reads columns by position. The order is day by day, oldest first, with the 4 features in table order within each day. Exact order:

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

Based on the `Activity Type` column.

| Group | Activity types | Feeds |
|---|---|---|
| Strength | Weight Training, Crossfit | `strength training`, `nr. sessions` |
| Alternative | Ride, Swim, Hike, Walk, Yoga, Virtual Ride, Workout | `hours alternative`, `nr. sessions` |
| Run | Every type containing "Run" (e.g. Run, Trail Run, Virtual Run) | `total km`, `nr. sessions` |
| Ignored | Any other type not listed above | nothing |
| Excluded | Any activity with `Commute` = True | nothing |

## 4. Not enough data

If the 7-day window contains **no run activity**, the app does not predict. It shows:

> Not enough data to calculate current injury risk.

## 5. Features left out of v0

| Training feature | Reason |
|---|---|
| `km Z3-4`, `km Z5-T1-T2` | They need a heart-rate zone approximation, and only about 53% of runs have HR. Candidate for v1. |
| `km sprinting` | No Strava source. |
| `perceived exertion` | Strava RPE is filled on 0.5% of runs, and its scale (1–10) differs from the training data (0–1). |
| `perceived trainingSuccess`, `perceived recovery` | No Strava source. |

## 6. Example

`example_row.csv` is a real row built from Dominik's export with event day 2026-10-15. Use it as stub input for the API and UI.

## 7. Sign-off

- [ ] D2 training preprocessing — can select these columns
- [ ] B2 model — can train on these columns
- [ ] A2 Strava features — can build these columns
- [ ] N2 API — expects these columns

*Contract version: v0*
