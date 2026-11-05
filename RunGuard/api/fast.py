from fastapi import FastAPI, UploadFile
import pandas as pd

from RunGuard.ml_logic.model import load_model, predict
from RunGuard.ml_logic.strava import build_features, prepare_activities

# FastAPI instance
app = FastAPI()

# Model_name chosen is the one I use locally
# Called before endpoint to only load once at startup, not at every call
model = load_model('model_20261105-144722.pkl')

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
