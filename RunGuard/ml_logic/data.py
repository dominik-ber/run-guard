import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def train_test_data_prep(data: pd.DataFrame):
    """Prepare training data for the ML model.

    Args:
        data: Raw data from the "day_approach_maskedID_timeseries.csv" file as Pandas DataFrame.

    Returns:
        X_train, y_train, X_test, y_test
    """

    columns = ['nr. sessions',
    'total km',
    'strength training',
    'hours alternative',
    'nr. sessions.1',
    'total km.1',
    'strength training.1',
    'hours alternative.1',
    'nr. sessions.2',
    'total km.2',
    'strength training.2',
    'hours alternative.2',
    'nr. sessions.3',
    'total km.3',
    'strength training.3',
    'hours alternative.3',
    'nr. sessions.4',
    'total km.4',
    'strength training.4',
    'hours alternative.4',
    'nr. sessions.5',
    'total km.5',
    'strength training.5',
    'hours alternative.5',
    'nr. sessions.6',
    'total km.6',
    'strength training.6',
    'hours alternative.6',
    'injury',
    'Athlete ID',
    'Date']

    #Select only relevant columns from raw data
    data = data[columns]

    #Simple cleaning: drop rows with NaN in any column
    n_before = len(data)
    data = data.dropna()
    print("rows dropped due to NaN:", n_before - len(data))
    assert not data.isna().any().any(), "NaNs left in data!"

    #Simple cleaning: all feature columns must be >= 0
    feature_cols = [c for c in data.columns if c not in ["injury", "Athlete ID", "Date"]]
    n_before = len(data)
    data = data[(data[feature_cols] >= 0).all(axis=1)]
    print("rows dropped due to negative values:", n_before - len(data))
    assert (data[feature_cols] >= 0).all().all(), "Negative values left in data!"

    #Simple cleaning: nr. sessions and strength training must be whole numbers
    int_cols = [c for c in data.columns if c.startswith(("nr. sessions", "strength training"))]
    n_before = len(data)
    data = data[(data[int_cols] % 1 == 0).all(axis=1)]
    data = data.astype({c: int for c in int_cols})
    print("rows dropped due to non-integer sessions / strength:", n_before - len(data))
    assert all(pd.api.types.is_integer_dtype(data[c]) for c in int_cols), "Non-int sessions / strength left in data!"

    #Athlete grouped train / test split with StratifiedGroupKFold
    X = data.drop(columns=["injury", "Athlete ID", "Date"])
    y = data["injury"]
    groups = data["Athlete ID"]

    # 5 folds -> one fold as test = ~20% of athletes
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X, y, groups))

    X_train, y_train, g_train = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
    X_test,  y_test,  g_test  = X.iloc[test_idx],  y.iloc[test_idx],  groups.iloc[test_idx]

    # Tests and checks
    assert set(g_train).isdisjoint(set(g_test)), "Athlete leakage!"
    print("length train/test: ", len(X_train), len(X_test))
    print("injuries  train/test:", y_train.sum(), y_test.sum())
    print("inj. rate train/test:", y_train.mean().round(4), y_test.mean().round(4))
    print("athletes  train/test:", g_train.nunique(), g_test.nunique())
    print("athlete IDs used in the train split:", set(g_train))
    print("athlete IDs used in the test split:", set(g_test))


    return X_train, y_train, X_test, y_test
