"""Train XGBoost without class weights, for the side-by-side comparison in the UI.
Current Ben's initial model trained with balancing.

Run from the repo root (the Dockerfile runs it right after init_model.py):

    python -m RunGuard.interface.init_xgb_no_weight

"""

import pickle
from pathlib import Path

import pandas as pd

from RunGuard.ml_logic import model
from RunGuard.ml_logic import train_test_data

MODEL_FILE_NAME = "xgb_no_weight.pkl"


def main():
    df = pd.read_csv("raw_data/day_approach_maskedID_timeseries.csv")

    # The same train and test athletes as the main model in init_model.py
    X_train, y_train, X_test, y_test = train_test_data.train_test_data_prep(df)

    xgb_model = model.initialize_xgboost_model(y_train)
    # No additional weight compraing to Ben's model.
    xgb_model.set_params(scale_pos_weight=1)

    # train_model returns two values: the fit history and the trained model.
    # We only need the trained model; the history is not used here.
    history, trained_model = model.train_model(xgb_model, X_train, y_train)

    print("Test set evaluation:")
    print(model.evaluate_model(trained_model, X_test, y_test))
    mean_probability = model.predict(trained_model, X_test).mean()
    print(f"Mean probability {mean_probability:.2%} (real injury rate {y_test.mean():.2%})")

    # Save into RunGuard/models/, the folder load_model() reads from.
    # save_model() always names the file model_<timestamp>.pkl, so we save it here ourselves.
    models_dir = Path(model.__file__).resolve().parents[1] / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / MODEL_FILE_NAME
    with open(model_path, "wb") as f:
        pickle.dump(trained_model, f)

    print(f"Model saved in : {model_path}")


if __name__ == "__main__":
    main()
