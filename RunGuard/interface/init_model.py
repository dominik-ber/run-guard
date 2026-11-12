from RunGuard.ml_logic import model
from RunGuard.ml_logic import train_test_data
import pandas as pd
from RunGuard.ml_logic.registry import mlflow_run
import  mlflow
from RunGuard.params import *
from sklearn.metrics import precision_recall_fscore_support
import os
import google.auth
import google.auth.transport.requests

def connexion():
    # Utilise le compte de service pour générer un jeton valide
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)

    # Injecte le token pour que MLflow l'utilise lors des appels réseau
    os.environ["MLFLOW_TRACKING_TOKEN"] = credentials.token

    return credentials.token




def train():
    df = pd.read_csv("raw_data/day_approach_maskedID_timeseries.csv")

    # Load training data
    X_train, y_train, X_test, y_test= train_test_data.train_test_data_prep(df)

    # Initialize the XGBoost model
    xgb_model = model.initialize_xgboost_model(y_train)
    #mlflow.log_params(xgb_model.get_params())

    # Train the model
    xgb_model.fit(X_train, y_train)
    model.save_model(xgb_model)




@mlflow_run
def main():
    """
    Main function to initialize and train the XGBoost model.
    """

    xgb_model = model.load_model()
    #y_pred = xgb_model.predict(X_test)
    #breakpoint()
    #precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    mlflow.autolog()
    mlflow.xgboost.log_model(
            xgb_model=xgb_model,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME
        )
    # metrics = {
    #     "precision": precision.tolist(),
    #     "recall": recall.tolist(),
    #     "f1_score": f1.tolist(),
    #     "support": support.tolist()
    # }

    #mlflow.log_metrics(metrics)

    # Save the trained model
    #model.save_model(trained_model)

    print("Model training and saving completed successfully.")
    return xgb_model


if __name__ == "__main__":
    connexion()
    main()
    #train()
