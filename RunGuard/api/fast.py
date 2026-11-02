from fastapi import FastAPI
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
def predict(
    nr_sessions,
    total_km,
    strength_training,
    hours_alternative,
    nr_sessions_1,
    total_km_1,
    strength_training_1,
    hours_alternative_1,
    nr_sessions_2,
    total_km_2,
    strength_training_2,
    hours_alternative_2,
    nr_sessions_3,
    total_km_3,
    strength_training_3,
    hours_alternative_3,
    nr_sessions_4,
    total_km_4,
    strength_training_4,
    hours_alternative_4,
    nr_sessions_5,
    total_km_5,
    strength_training_5,
    hours_alternative_5,
    nr_sessions_6,
    total_km_6,
    strength_training_6,
    hours_alternative_6
):
    # Use the model function in the package to run the prediction
    prediction = model(
        nr_sessions,
        total_km,
        strength_training,
        hours_alternative,
        nr_sessions_1,
        total_km_1,
        strength_training_1,
        hours_alternative_1,
        nr_sessions_2,
        total_km_2,
        strength_training_2,
        hours_alternative_2,
        nr_sessions_3,
        total_km_3,
        strength_training_3,
        hours_alternative_3,
        nr_sessions_4,
        total_km_4,
        strength_training_4,
        hours_alternative_4,
        nr_sessions_5,
        total_km_5,
        strength_training_5,
        hours_alternative_5,
        nr_sessions_6,
        total_km_6,
        strength_training_6,
        hours_alternative_6
    )
    return {'prediction': f"{prediction}%"}
