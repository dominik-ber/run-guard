from RunGuard.ml_logic import model
from RunGuard.ml_logic import train_test_data
import pandas as pd
from RunGuard.ml_logic.registry import mlflow_run
import  mlflow
from RunGuard.params import *
from sklearn.metrics import precision_score, recall_score, f1_score,average_precision_score
import os
import google.auth
import google.auth.transport.requests
from sklearn.model_selection import train_test_split


def connexion():
    # Utilise le compte de service pour générer un jeton valide
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)

    # Injecte le token pour que MLflow l'utilise lors des appels réseau
    os.environ["MLFLOW_TRACKING_TOKEN"] = credentials.token

    return credentials.token

def prepare_data():
    df = pd.read_csv("raw_data/day_approach_maskedID_timeseries.csv")

    # Load training and test data (test = held-out athletes)
    X_train, y_train, X_test, y_test = train_test_data.train_test_data_prep(df)

    return X_train, y_train, X_test, y_test



def train(X_train, y_train, X_val, y_val):
    # Initialize the XGBoost model
    xgb_model = model.initialize_xgboost_model(y_train)
    mlflow.log_params(xgb_model.get_params())

    # Train the model
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],)
    return xgb_model


@mlflow_run
def main_mlflow():
    """
    Main function to initialize and train the XGBoost model in mmlflow context
    """
    connexion()
    X_train, y_train, X_test, y_test = prepare_data()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42,
    )
    xgb_model = train(X_tr, y_tr, X_val, y_val)
    print("Test set evaluation: ok ")

    y_pred = xgb_model.predict(X_test)
    #breakpoint()

    mlflow.autolog()
    print("Autolog : ok ")
    mlflow.xgboost.log_model(
            xgb_model=xgb_model,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME
        )

    print("Log model : ok ")

    mlflow.log_metrics({
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "n_positive": float((y_test == 1).sum()),
    })

    mlflow.log_metrics({
        "best_iteration": float(xgb_model.best_iteration),
        "pr_auc_val": float(average_precision_score(y_val, xgb_model.predict_proba(X_val)[:, 1])),
        "pr_auc_test": float(average_precision_score(y_test, xgb_model.predict_proba(X_test)[:, 1])),
    })


    print("Log metrics : ok ")
    # Save the trained model
    model.save_model(xgb_model)

    print("Model training and saving completed successfully.")
    return xgb_model



def main():
    """
    Main function to initialize and train the XGBoost model.
    """

    df = pd.read_csv("raw_data/day_approach_maskedID_timeseries.csv")

    # Load training and test data (test = held-out athletes)
    X_train, y_train, X_test, y_test = train_test_data.train_test_data_prep(df)

    # Initialize the XGBoost model
    xgb_model = model.initialize_xgboost_model(y_train)

    # Train the model
    _, trained_model = model.train_model(xgb_model, X_train, y_train)

    # Evaluate on the test athletes against the no-skill baseline
    print("Test set evaluation:")
    print(model.evaluate_model(trained_model, X_test, y_test))

    # Save the trained model
    model.save_model(trained_model)

    print("Model training and saving completed successfully.")


if __name__ == "__main__":
    main_mlflow()
