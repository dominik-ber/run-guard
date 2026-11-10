from fastapi import FastAPI, HTTPException, UploadFile
import pandas as pd
import requests

from RunGuard.ml_logic.garmin import fetch_activities, process_data
from RunGuard.ml_logic.model import load_model, predict
from RunGuard.ml_logic.strava import build_features, prepare_activities

# FastAPI instance
app = FastAPI()

# Called before endpoint to only load once at startup, not at every call
model = load_model()

# Health endpoint
@app.get("/health")
def health():
    return {'status': "ok"}

# Prediction endpoint
@app.post("/predict")
def predict_endpoint(file: UploadFile):

    # Accept the uploaded file and store in pd df
    raw = pd.read_csv(file.file, low_memory=False)

    # Clean the data and build a row for prediction
    activities = prepare_activities(raw)
    row = build_features(activities)

    # Use the predict function in the package to run the prediction
    prediction = predict(model, row)

    return {'probability': float(prediction[0])}


# Garmin prediction endpoint: fetches live, nothing is stored on disk
@app.get("/predict/garmin")
def predict_garmin():

    # 14 days = the 7-day window plus margin for activities spanning midnight
    try:
        activities = fetch_activities(days=14)
    except RuntimeError as e:  # API key missing
        raise HTTPException(status_code=500, detail=str(e))
    except requests.RequestException as e:  # Garmin unreachable / rejected
        raise HTTPException(status_code=502, detail=f"Garmin request failed: {e}")

    # Event day = today, so the window is the last 7 full days
    row = process_data(activities, event_day=pd.Timestamp.today().normalize())
    prediction = predict(model, row)

    return {'probability': float(prediction[0])}
