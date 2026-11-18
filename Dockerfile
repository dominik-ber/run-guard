FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


COPY RunGuard ./RunGuard

COPY raw_data ./raw_data

# Training during the build : produces /app/models/model.pkl
RUN python -m RunGuard.interface.init_model
# Second XGBoost without class weights, for the comparison in the UI: produces RunGuard/models/xgb_no_weight.pkl
# (the MLP is not trained here: its weights come ready in RunGuard/mlp_weights/, see train_mlp.py)
RUN python -m RunGuard.interface.init_xgb_no_weight


CMD uvicorn RunGuard.api.fast:app --host 0.0.0.0 --port ${PORT:-8000}
