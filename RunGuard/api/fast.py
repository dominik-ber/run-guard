from fastapi import FastAPI
import pickle
import pandas as pd

from RunGuard.ml_logic.model import predict
from RunGuard.ml_logic.strava import build_features, clean_activities
from RunGuard.ml_logic.model import load_model

# FastAPI instance
app = FastAPI()

# Health endpoint
@app.get("/health")
def health():
    return {'status': "ok"}

# Prediction endpoint
@app.get("/predict")
def predict_endpoint(file):

    activities = clean_activities(file)
    row = build_features(activities)
    # model_name chosen is placeholder
    model = load_model('model_20230101-123456.pkl')
    # Use the predict function in the package to run the prediction
    prediction = predict(model, row)

    return {'prediction': f"{prediction[0] * 100:.1f}%"}
