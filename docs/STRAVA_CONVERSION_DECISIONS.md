# Strava → Feature Contract: conversion decisions

This note records the decisions taken while preparing the app-side conversion (A2) of a Strava `activities.csv` export into the columns defined in [`FEATURE_CONTRACT.md`](../FEATURE_CONTRACT.md) (v0).

All numbers below come from Dominik's full Strava export: 2,727 activities from April 2014 to October 2026, of which 1,872 are runs.

## Summary

| # | Topic | Decision | Affects |
|---|---|---|---|
| # | Topic | Decision | Affects | Status |
|---|---|---|---|---|
| 1 | Time zone of `Activity Date` | Keep UTC as exported, no conversion | day boundaries of every feature | agreed with Dominik |
| 2 | Warm-ups and cool-downs recorded as separate runs | Remove runs shorter than 2 km completely | `nr. sessions`, `total km` | proposed by Dominik |
| 3 | Two `Distance` columns, one with mixed units | Use only `Distance.1` (metres) and convert to km; ignore `Distance` | `total km` | to be agreed |
| 4 | Activities that run past midnight (UTC) | Spread their moving time and distance evenly over their elapsed time, then split by day | `total km`, `hours alternative` | to be agreed |

Decisions 2, 3 and 4 change or extend rules written in the contract; the contract still has to be updated.

## 1. Time zone

`Activity Date` is exported in UTC. Run names confirm it: "Lunch Run" activities start around 10:00 and "Evening Run" activities around 17:00 in the export, which is 12:00 and 19:00 in Berlin.

**Decision:** keep UTC and do not convert to local time.

**Known consequence:** for a runner in Central Europe (UTC+1 in winter, UTC+2 in summer), an activity that starts shortly after local midnight falls on the previous UTC day. This only concerns activities starting between 00:00 and 01:00 (winter) or 02:00 (summer) local time.

## 2. Removing short runs (warm-ups and cool-downs)

A warm-up is an easy run before the main workout (intervals, tempo run, race); a cool-down is an easy run after it. Both belong to the same training session. Strava does not split them off: Dominik starts and stops a separate recording on his watch for them, and later names them by hand.

Example from the export, 9 August 2025:

| Start (UTC) | km | min | Part |
|---|---|---|---|
| 06:49 | 0.81 | 4 | warm-up |
| 07:01 | 5.02 | 19 | main part (5 km race) |
| 07:28 | 1.78 | 10 | cool-down |

Counted as activities this day has 3 sessions (2 after the contract's cap). For the coach and for the training data it is 1 session.

**Decision (proposed by Dominik):** remove every run shorter than 2 km from the conversion completely. Removed runs count neither as a session nor towards `total km`.

On the example day this leaves one session of 5.02 km.

Effect on the export:

| | |
|---|---|
| Runs removed | 193 of 1,872 |
| Distance removed | 198 km of 15,352 km (1.3%) |
| Days that lose all their runs | 13 |

Notes for the team:

- **Short runs are not only warm-ups and cool-downs.** Of the 193 removed runs, 49 are named as such; the other 144 carry default names ("Morning Run" 94, "Evening Run" 14, "Lunch Run" 10, ...). These may be unnamed warm-ups or genuinely short runs; both are removed.
- **Long warm-ups and cool-downs stay.** 27 runs named "warm-up" or "cool-down" are 2 km or longer and remain separate sessions.
- **Some split workouts still count twice.** After the filter, 43 pairs of runs on the same day start less than 30 minutes after the previous one ends. They are counted as 2 sessions.
- **The distance of warm-ups and cool-downs is lost** from `total km` (1.3% of all running distance in this export).

An earlier proposal merged runs on the same day less than 30 minutes apart into one session and kept their distance. It was replaced by this simpler rule.

## 3. Distance in kilometres

The raw CSV has two columns named `Distance`. pandas renames the second one to `Distance.1` when reading the file (the same happens to `Elapsed Time`, `Max Heart Rate`, `Relative Effort` and `Commute`). The first block of columns is formatted for display, the second holds raw values in base units:

| Column (as read by pandas) | Unit | Example |
|---|---|---|
| `Distance` | text; km for most activity types, **metres for swims**, with `,` as thousands separator; rounded to 2 decimals | `8.36` (run), `2,100` (swim) |
| `Distance.1` | numeric, always metres, not rounded | `8360.2` |

Both describe the same distance, so one is enough. `Distance` is the one to drop: reading it as kilometres works for runs but silently breaks for swims, and it is less precise.

**Decision:** use only `Distance.1` and convert it to kilometres; ignore `Distance`.

```python
activities["km"] = raw["Distance.1"] / 1000  # "Distance.1" is always metres
```

Checked on all 2,727 activities of the export:

- `Distance.1` is filled for every activity;
- `Distance` and `Distance.1` describe the same distance for every activity (no difference above 10 m once units are taken into account);
- for runs, daily totals from `Distance.1` differ from totals of the rounded `Distance` by at most 0.03 km;
- median distance per type is plausible: Run 6.91 km, Swim 1.70 km, Ride 49.69 km, and 0 km for Weight Training, Yoga and Crossfit.

The app will most likely read activities from the Strava API rather than from a CSV export. The API also returns distance in metres, matching `Distance.1`.

## 4. Splitting activities that run past midnight

Strava stores every activity as one record dated by its start. Without a rule, the whole activity lands on its start day, however long it lasted. The export contains ultra-distance rides that last more than a day:

| Start (UTC) | Activity | Moving h | Elapsed h | km |
|---|---|---|---|---|
| 2018-06-02 05:01 | Ride "Brevet" | 22.4 | 28.1 | 612 |
| 2019-05-25 06:13 | Ride "grüß gott" | 27.0 | 34.9 | 627 |
| 2019-08-18 17:28 | Ride "Paris-Brest-Paris" | 55.9 | 89.7 | 1,217 |

Booked on the start day, Paris-Brest-Paris alone gives 55.9 hours of alternative training in one day. That is physically impossible and far above the training data's maximum of 20 hours.

These are cycling brevets: long-distance rides with checkpoints, where riders sleep a few hours and stop to eat. Their elapsed time is 1.2 to 1.6 times their moving time (1.6 for Paris-Brest-Paris, with 33.8 hours of breaks over almost 4 days), and their average moving speed is a plausible 22 to 27 km/h.

**Decision:** spread an activity evenly over the time it covers, and split it across the UTC days in that time.

- **The time an activity covers is its `Elapsed Time`**, from start to finish including breaks. Each day gets the share of that time that falls inside it, and the same share of the moving time and of the distance. Riding and breaks are spread together.
- **Exception: a recording left running.** If `Elapsed Time` is more than twice `Moving Time`, the activity is treated as covering only its `Moving Time` from the start. One ride in the export has 0.3 moving hours but 1,859 elapsed hours (78 days), another 2.4 moving hours but 842 elapsed hours. Spread over their elapsed time, these short rides would cover months. All brevets stay well below the factor of 2.
- **Sessions are counted once, on the start day.** One activity is still one session, however many days it covers.
- `strength training` is a count and is not split.

Effect on the export: 16 activities are split (11 runs, 4 rides, 1 swim). The rides:

| Activity | Hours per day | km per day |
|---|---|---|
| Paris-Brest-Paris | 4.1 / 15.0 / 15.0 / 15.0 / 7.0 | 89 / 326 / 326 / 326 / 152 |
| grüß gott | 13.8 / 13.2 | 319 / 307 |
| Brevet | 15.1 / 7.3 | 414 / 199 |
| coolyo | 15.5 / 1.7 | 375 / 40 |

The highest daily value of `hours alternative` is now 15.5, within the training data's range (maximum 20). About 15 riding hours and 326 km per full day is a typical pace on such brevets.

An earlier version spread only the moving time from the start, ignoring breaks. It gave Paris-Brest-Paris 6.5 / 24.0 / 24.0 / 1.4 hours: two days of riding around the clock, above the training data's range.

**Known limitations:**

- Riding and breaks are assumed to be spread evenly over the activity. The real split (for example, more riding on the first day, sleep at night) is only available from the per-second data in the activity's `.fit` file.
- The factor of 2 is a judgement call. In the export, 156 activities have elapsed time above twice moving time, almost all short rides that end on their start day either way.
- Short runs that start just before UTC midnight are split too. In the export these are 11 morning runs during trips in Asia (local time is several hours ahead of UTC), so a few kilometres move to the next UTC day. This follows from decision 1 (UTC).

## Next steps

1. Agree on decisions 3 and 4 with the contract owner, then update `FEATURE_CONTRACT.md` for decisions 2–4 (`nr. sessions`, `total km` and `hours alternative` rows).
2. Validate the converter against `example_row.csv` (event day 2026-10-15). The code is in `RunGuard/preprocessing/strava_clean.py` (`clean_activities`) and `RunGuard/features/strava.py` (`build_features`); `notebooks/strava_to_contract.ipynb` runs them on the export.
3. Confirm with the team that the Strava modules belong in new `RunGuard/preprocessing/` and `RunGuard/features/` subpackages, next to the project skeleton's `api/`, `interface/` and `ml_logic/`.
