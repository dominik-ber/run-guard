from fastapi import FastAPI
from RunGuard.ml_logic.strava import build_features, clean_activities
import pickle
import pandas as pd

from ml_logic.model import predict

# FastAPI instance
app = FastAPI()

# Health endpoint
@app.get("/health")
def health():
    return {'status': "ok"}

# Prediction endpoint
@app.get("/predict")
def predict(file):
    from RunGuard.ml_logic.model import load_model
    activities = clean_activities(file)
    row = build_features(activities)
    # model_name chosen is placeholder
    model = load_model('model_20230101-123456.pkl')
    # Use the predict function in the package to run the prediction
    prediction = predict(model, row)

    return {'prediction': f"{prediction}%"}
