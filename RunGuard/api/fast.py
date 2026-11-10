from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile
import pandas as pd
from pydantic import BaseModel
import requests

from RunGuard.ml_logic.garmin import GarminAuthError, fetch_activities, process_data
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


# Request body for the Garmin endpoint. POST body, not a URL parameter,
# so the key never ends up in server or proxy logs.
class GarminRequest(BaseModel):
    api_key: Optional[str] = None  # empty -> the server's own (demo) key


# Garmin prediction endpoint: fetches live, nothing is stored on disk
@app.post("/predict/garmin")
def predict_garmin(body: GarminRequest):

    api_key = (body.api_key or "").strip() or None

    # 14 days = the 7-day window plus margin for activities spanning midnight.
    # Error messages are fixed text on purpose: never echo anything key-related.
    try:
        activities = fetch_activities(days=14, api_key=api_key)
    except GarminAuthError:
        raise HTTPException(status_code=401, detail="Invalid Garmin API key.")
    except RuntimeError:  # no key given and no server key configured
        raise HTTPException(status_code=400, detail="Please enter your Garmin API key.")
    except requests.RequestException:  # Garmin unreachable or returned an error
        raise HTTPException(status_code=502, detail="Garmin request failed, please try again later.")

    # Event day = today, so the window is the last 7 full days
    row = process_data(activities, event_day=pd.Timestamp.today().normalize())
    prediction = predict(model, row)

    return {'probability': float(prediction[0])}
