from fastapi import FastAPI
from RunGuard.ml_logic.strava import build_features, clean_activities
import pickle


### from ml_logic.model import <INSERT MODEL NAME>

# FastAPI instance
app = FastAPI()

# Health endpoint
@app.get("/health")
def health():
    return {'status': "ok"}

# Prediction endpoint
@app.get("/predict")
def predict(activities_csv):
    activities = clean_activities(activities_csv)
    row = build_features(activities)
    # Use the model function in the package to run the prediction
    prediction = model(row)

    return {'prediction': f"{prediction}%"}
