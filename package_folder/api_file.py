from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from package_folder.run_guard import my_prediction_function
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    return dict(greeting="Hello")

@app.get("/predict")
def predict_injurie():
    prediction = my_prediction_function()
    return {"prediction": prediction}
